import re
from collections import OrderedDict
from functools import partial
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fairseq.modules.moving_average_gated_attention import MovingAverageGatedAttention
from fairseq.modules.gated_cross_attention import GatedCrossAttention
from flash_attn.layers.patch_embed import PatchEmbed
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
from timm.models.helpers import named_apply
from torch import Tensor
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth

from activations.gated_4d import Gated4d, Gated4dV2
from activations.gated_mish import GatedMish
from activations.geglu import Geglu1dLearnable
from activations.gmm2d import GMMActivation2DFullyLearnable
from activations.hyperbolic_paraboloid import HyperbolicParaboloidActivation
from activations.mli2d import GatedMLISoftLut2Layer, GatedMLISoftLut2LayerWithLearnableCoords
from activations.raf import Raf2dFirstDegree, Raf2dSecondDegree, Raf2dThirdDegree
from model.mlp import GatedMlp

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
except ImportError:
    layer_norm_fn, RMSNorm = None, None

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn
except ImportError:
    layer_norm_fn = None


def create_mixer_cls(num_heads, qkv_bias, attn_drop, use_flash_attn, fused_bias_fc, use_mega=False, cross_attn=False, mega_z_dim=256, mega_hidden_dim=382, mega_n_dim=16, mega_hidden_dropout=0.1, mega_dropout=0.3, mega_truncation_length=1024, mega_rel_pos_bias='simple', mega_chunk_size=-1, mega_max_source_positions=1024, mega_act='silu'):
    if use_mega:
        if cross_attn:
            mixer_cls = partial(GatedCrossAttention,
            zdim=mega_z_dim,
            ndim=mega_n_dim,
            dropout=mega_dropout,
            attention_dropout=attn_drop,
            hidden_dropout=mega_hidden_dropout,
            activation=mega_act,
            attention_activation='softmax',
            norm_type='layernorm',
            prenorm=True,
            feature_dropout=False,
            rel_pos_bias='simple',
            max_positions=mega_max_source_positions,
        )
        else:
            mixer_cls = partial(MovingAverageGatedAttention,
                zdim=mega_z_dim,
                hdim=mega_hidden_dim,
                ndim=mega_n_dim,
                dropout=mega_dropout,
                attention_dropout=attn_drop,
                hidden_dropout=mega_hidden_dropout,
                chunk_size=mega_chunk_size,
                truncation=mega_truncation_length,
                rel_pos_bias=mega_rel_pos_bias,
                max_positions=mega_max_source_positions,
                activation=mega_act,
                attention_activation='softmax',
                bidirectional=True,
                norm_type='layernorm',
                prenorm=True,
                feature_dropout=False,
            )
    else:

        mixer_cls = partial(
            MHA,
            num_heads=num_heads,
            cross_attn=cross_attn,
            qkv_proj_bias=qkv_bias,
            dropout=attn_drop,
            fused_bias_fc=fused_bias_fc,
            use_flash_attn=use_flash_attn,
        )
    return mixer_cls


def create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_mlp):
    inner_dim = int(embed_dim * mlp_ratio)
    if not fused_mlp:
        if act_layer in ['glu', 'swiglu', 'geglu', 'gated-mish', 'gated-4d', 'gated-4dv2',
                                          'geglu-learnable', 'hyperbolic-paraboloid', 'gmm2d-gated', 'mli2d-gated',
                                          'mli2d-gated-learned-coords', 'raf2d-1degree', 'raf2d-2degree',
                                          'raf2d-3degree']:

            if act_layer == 'mli2d-gated':
                activation = GatedMLISoftLut2Layer(dim=int(embed_dim * 8 / 3))
            elif act_layer == 'gated-4d':
                activation = Gated4d()
            elif act_layer == 'gated-4dv2':
                activation = Gated4dV2()
            elif act_layer == 'gmm2d-gated':
                activation = GMMActivation2DFullyLearnable(dim=int(embed_dim * 8 / 3))
            elif act_layer == 'geglu-learnable':
                activation = Geglu1dLearnable()
            elif act_layer == 'hyperbolic-paraboloid':
                activation = HyperbolicParaboloidActivation(dim=int(embed_dim * 8 / 3))
            elif act_layer == 'gated-mish':
                activation = GatedMish()
            elif act_layer == 'mli2d-gated-learned-coords':
                activation = GatedMLISoftLut2LayerWithLearnableCoords(dim=int(embed_dim * 8 / 3))
            elif act_layer == 'raf2d-1degree':
                activation = Raf2dFirstDegree()
            elif act_layer == 'raf2d-2degree':
                activation = Raf2dSecondDegree()
            elif act_layer == 'raf2d-3degree':
                activation = Raf2dThirdDegree()
            else:
                activation = (F.sigmoid if act_layer == 'glu' else (F.silu if act_layer == 'swiglu' else F.gelu))
            mlp_cls = partial(
                GatedMlp,
                hidden_features=int(embed_dim * 8 / 3),
                activation=activation,
            )
        else:
            mlp_cls = partial(Mlp, hidden_features=inner_dim, activation=act_layer())
    else:
        mlp_cls = partial(FusedMLP, hidden_features=inner_dim, activation=act_layer)
    return mlp_cls

class MegaBlock(Block):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm, dropout_cls=nn.Dropout, prenorm=True,
                 resid_dropout1=0.0, resid_dropout2=0.0, drop_path1=0.0, drop_path2=0.0, fused_dropout_add_ln=False,
                 return_residual=False, residual_in_fp32=False, sequence_parallel=False, mark_shared_params=False, use_mega=False):
        super().__init__(dim, mixer_cls, mlp_cls, norm_cls, dropout_cls, prenorm, resid_dropout1, resid_dropout2,
                         drop_path1, drop_path2, fused_dropout_add_ln, return_residual, residual_in_fp32,
                         sequence_parallel, mark_shared_params)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        mixer_subset=None,
        mixer_kwargs=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        # TODO: input to mixer is of shape: seq x batch x embed
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path1(self.dropout1(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm1.weight,
                    self.norm1.bias,
                    residual=residual,
                    eps=self.norm1.eps,
                    dropout_p=self.dropout1.p if self.training else 0.0,
                    rowscale=rowscale1,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm1, RMSNorm)
                )

            hidden_states = einops.rearrange(hidden_states, 'b s hid -> s b hid')
            if mixer_subset is not None:
                # cross attention
                hidden_states = self.mixer(hidden_states[mixer_subset, :, :], hidden_states, hidden_states)
            else:
                hidden_states = self.mixer(hidden_states)
            if isinstance(hidden_states, tuple):
                # For cases when mixer returns tuples (like MEGA) of output and residual we just need the first argument
                hidden_states = hidden_states[0]
                hidden_states = einops.rearrange(hidden_states, 's b hid -> b s hid')
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.drop_path2(self.dropout2(hidden_states))
                    residual = (dropped + residual) if residual is not None else dropped
                    hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(
                            torch.ones(
                                hidden_states.shape[:-1],
                                device=hidden_states.device,
                                dtype=hidden_states.dtype,
                            )
                        )
                    hidden_states, residual = layer_norm_fn(
                        hidden_states,
                        self.norm2.weight,
                        self.norm2.bias,
                        residual=residual,
                        eps=self.norm2.eps,
                        dropout_p=self.dropout2.p if self.training else 0.0,
                        rowscale=rowscale2,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        is_rms_norm=isinstance(self.norm2, RMSNorm)
                    )
                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            hidden_states = einops.rearrange(hidden_states, 'b s hid -> s b hid')
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm1(
                    (self.drop_path1(self.dropout1(mixer_out)) + hidden_states).to(
                        dtype=self.norm1.weight.dtype
                    )
                )
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(
                        torch.ones(
                            mixer_out.shape[:-1], device=mixer_out.device, dtype=mixer_out.dtype
                        )
                    )
                hidden_states = layer_norm_fn(
                    mixer_out,
                    self.norm1.weight,
                    self.norm1.bias,
                    residual=hidden_states,
                    eps=self.norm1.eps,
                    dropout_p=self.dropout1.p if self.training else 0.0,
                    rowscale=rowscale1,
                    prenorm=False,
                    is_rms_norm=isinstance(self.norm1, RMSNorm)
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                if not self.fused_dropout_add_ln:
                    hidden_states = self.norm2(
                        (self.drop_path2(self.dropout2(mlp_out)) + hidden_states).to(
                            dtype=self.norm2.weight.dtype
                        )
                    )
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(
                            torch.ones(
                                mlp_out.shape[:-1], device=mlp_out.device, dtype=mlp_out.dtype
                            )
                        )
                    hidden_states = layer_norm_fn(
                        mlp_out,
                        self.norm2.weight,
                        self.norm2.bias,
                        residual=hidden_states,
                        eps=self.norm2.eps,
                        dropout_p=self.dropout2.p if self.training else 0.0,
                        rowscale=rowscale2,
                        prenorm=False,
                        is_rms_norm=isinstance(self.norm2, RMSNorm)
                    )
            return hidden_states


def create_block(
    embed_dim,
    num_heads,
    mlp_ratio,
    qkv_bias,
    drop_rate,
    attn_drop_rate,
    drop_path1,
    drop_path2,
    norm_layer,
    act_layer,
    use_flash_attn,
    fused_bias_fc,
    fused_mlp,
    fused_dropout_add_ln,
    layer_idx=None,
    n_layer=None,
    last_layer_subset=False,
    use_mega=False
):
    mixer_cls = create_mixer_cls(
        num_heads,
        qkv_bias,
        attn_drop_rate,
        use_flash_attn,
        fused_bias_fc,
        use_mega=use_mega,
        cross_attn=(last_layer_subset and layer_idx == n_layer - 1),
    )
    mlp_cls = create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_mlp)
    # TD [2022-10-15]: Force residual in fp32 in case of DeepSpeed
    bloc_cls = MegaBlock if use_mega else Block

    block = bloc_cls(
        embed_dim,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_layer,
        prenorm=True,
        resid_dropout1=drop_rate,
        resid_dropout2=drop_rate,
        drop_path1=drop_path1,
        drop_path2=drop_path2,
        fused_dropout_add_ln=fused_dropout_add_ln,
        residual_in_fp32=True,
    )
    return block


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        use_flash_attn=False,
        fused_bias_fc=False,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        use_mega=False
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool == "token", "Only support pooling with CLS token"
        assert class_token
        assert init_values is None, "LayerScale is not supported yet"
        assert weight_init == ""
        assert fc_norm is None
        # pre_norm seems redundant, as there's a LayerNorm right at the start of each block, idk
        assert not pre_norm
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class

        patch_embed_extra_kwargs = (
            {"fused_bias_fc": fused_bias_fc} if embed_layer is PatchEmbed else {}
        )
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            **patch_embed_extra_kwargs,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.blocks = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    drop_path1=dpr[i - 1] if i > 0 else 0.0,
                    drop_path2=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_flash_attn=use_flash_attn,
                    fused_bias_fc=fused_bias_fc,
                    fused_mlp=fused_mlp,
                    fused_dropout_add_ln=fused_dropout_add_ln,
                    layer_idx=i,
                    n_layer=depth,
                    last_layer_subset=(global_pool == "token"),
                    use_mega=use_mega
                )
                for i in range(depth)
            ]
        )

        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_path = StochasticDepth(p=dpr[-1], mode="row")
        self.norm = norm_layer(embed_dim)

        self.fused_dropout_add_ln = fused_dropout_add_ln
        if self.fused_dropout_add_ln and layer_norm_fn is None:
            raise ImportError("Triton is not installed")

        # Classifier Head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode == ""
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return x

    def forward_features(self, x, all_tokens=True):
        """
        If all_tokens==False and self.global_pool == 'token', we only return the features for the
        cls token.
        """
        x = self.patch_embed(x)
        hidden_states = self._pos_embed(x)
        residual = None
        if self.global_pool != "token" or all_tokens:
            # if True:
            for block in self.blocks:
                hidden_states, residual = block(hidden_states, residual)
        else:
            for block in self.blocks[:-1]:
                hidden_states, residual = block(hidden_states, residual)
            # For the last layer, we only want the 1st token of the output. So we do cross-attention
            # where the query is the 1st token and the key/value is the whole sequence.
            hidden_states, residual = self.blocks[-1](
                hidden_states, residual, mixer_subset=slice(0, 1)
            )
        if not self.fused_dropout_add_ln:
            residual = self.drop_path(self.dropout(hidden_states)) + residual
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            if self.drop_path.p == 0 or not self.training:
                rowscale = None
            else:
                rowscale = self.drop_path(
                    torch.ones(
                        hidden_states.shape[:-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )
            # Set prenorm=False here since we don't need to the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                eps=self.norm.eps,
                dropout_p=self.dropout.p if self.training else 0.0,
                rowscale=rowscale,
                prenorm=False,
            )
        return hidden_states

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x, all_tokens=False)
        x = self.forward_head(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        patch_embed_weight = state_dict["patch_embed.proj.weight"]
        if patch_embed_weight.dim() == 4:
            # convert from Conv2d to Linear
            state_dict["patch_embed.proj.weight"] = rearrange(
                patch_embed_weight, "o c h w -> o (c h w)"
            )

        def key_mapping_attn(key):
            key = re.sub(r"^blocks.(\d+).attn.qkv.", r"blocks.\1.mixer.Wqkv.", key)
            key = re.sub(r"^blocks.(\d+).attn.proj.", r"blocks.\1.mixer.out_proj.", key)
            return key

        state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
        n_layer = len(self.blocks)
        # Convert from Wqkv to Wq and Wkv for cross attention (last layer)
        if (
            self.blocks[-1].mixer.cross_attn
            and f"blocks.{n_layer - 1}.mixer.Wqkv.weight" in state_dict
        ):
            Wqkv = state_dict.pop(f"blocks.{n_layer - 1}.mixer.Wqkv.weight")
            bqkv = state_dict.pop(f"blocks.{n_layer - 1}.mixer.Wqkv.bias")
            state_dict[f"blocks.{n_layer - 1}.mixer.Wq.weight"] = Wqkv[: self.embed_dim]
            state_dict[f"blocks.{n_layer - 1}.mixer.Wkv.weight"] = Wqkv[self.embed_dim :]
            state_dict[f"blocks.{n_layer - 1}.mixer.Wq.bias"] = bqkv[: self.embed_dim]
            state_dict[f"blocks.{n_layer - 1}.mixer.Wkv.bias"] = bqkv[self.embed_dim :]
        return super().load_state_dict(state_dict, strict=strict)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def vit_base_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    assert not pretrained
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model

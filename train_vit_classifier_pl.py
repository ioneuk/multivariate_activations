import argparse
import os

import torch.nn.functional as F
import lightning as L
import torch
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10

from model.vit import VisionTransformer
from tasks.image_classification import VITLightningWrapper

L.seed_everything(7)

PATH_DATASETS = os.environ.get('PATH_DATASETS', '/home/user/datasets/cifar10')
LR = 1e-4
BATCH_SIZE = 16
EFFECTIVE_BATCH_SIZE = 256
ACCUMULATE_GRAD_BATCHES = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
assert ACCUMULATE_GRAD_BATCHES * BATCH_SIZE == EFFECTIVE_BATCH_SIZE
NUM_WORKERS = 1


class VITLightningWrapper(L.LightningModule):
    def __init__(self, size, batch_size, lr=1e-4, activation_fn_name = 'raf2d-1degree'):
        super().__init__()

        self.save_hyperparameters()

        self.model = VisionTransformer(img_size=size, num_classes=10, act_layer=activation_fn_name)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task='multiclass', num_classes=10)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        steps_per_epoch = 45000 // self.hparams.batch_size
        scheduler_dict = {
            'scheduler': OneCycleLR(
                optimizer,
                1e-3,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


# %%
def train(args):
    def split_dataset(dataset, val_split=0.2, train=True):
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = get_splits(len_dataset, val_split)
        dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))

        if train:
            return dataset_train
        return dataset_val

    def get_splits(len_dataset, val_split):
        """Computes split lengths for train and validation set."""
        if isinstance(val_split, int):
            train_len = len_dataset - val_split
            splits = [train_len, val_split]
        elif isinstance(val_split, float):
            val_len = int(val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f'Unsupported type {type(val_split)}')

        return splits

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.Resize(args.img_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset_train = CIFAR10(PATH_DATASETS, train=True, download=True, transform=train_transforms)
    dataset_val = CIFAR10(PATH_DATASETS, train=True, download=True, transform=test_transforms)
    dataset_train = split_dataset(dataset_train)
    dataset_val = split_dataset(dataset_val, train=False)
    dataset_test = CIFAR10(PATH_DATASETS, train=False, download=True, transform=test_transforms)

    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = VITLightningWrapper(size=args.img_size, batch_size=BATCH_SIZE, lr=LR)
    wandb_logger = WandbLogger(project='cifar10-classification')

    trainer = L.Trainer(
        max_epochs=30,
        accelerator='auto',
        devices=args.n_devices,
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval='step')],
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        strategy='ddp_spawn'
    )

    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_devices', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=384)
    train(parser.parse_args())

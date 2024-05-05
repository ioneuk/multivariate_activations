import torch
import triton
import triton.language as tl


@triton.jit
def gmm2d(x, weights, output, SEQ_LEN: tl.constexpr, HIDDEN: tl.constexpr):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_half_idx = tl.program_id(2)

    pi = 3.141592653589793
    two_pi = 2 * pi

    idx_base = batch_idx * SEQ_LEN * HIDDEN + seq_idx * HIDDEN + hidden_half_idx * 2
    
    x_0 = tl.load(x + idx_base)
    x_1 = tl.load(x + idx_base + 1)

    r_0 = 0.0
    r_1 = 0.0

    mean_0_0, mean_0_1 = 0.0, 0.0
    mean_1_0, mean_1_1 = 0.0, 1.0
    mean_2_0, mean_2_1 = 1.0, 0.0
    mean_3_0, mean_3_1 = 1.0, 1.0

    ivc00, ivc01 = 0.4453, 0.0057
    ivc10, ivc11 = 0.4465, -0.0024
    ivc20, ivc21 = 0.4457, -0.0034
    ivc30, ivc31 = 0.4466, 0.0025

    det0, det1, det2, det3 = 5.0428, 5.0161, 5.0334, 5.0148
    
    # Compute Gaussian component 0
    d_0 = x_0 - mean_0_0
    d_1 = x_1 - mean_0_1
    sigma_d_0 = d_0 * ivc00 + d_1 * ivc01
    sigma_d_1 = d_0 * ivc01 + d_1 * ivc00
   
    matmul_0 = -0.5 * d_0 * sigma_d_0
    matmul_1 = -0.5 * d_1 * sigma_d_1
    gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det0) * two_pi)
    # tl.store(gaussians + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN*2 + hidden_half_idx*2, gaussian_pdf)
    w_0 = tl.load(weights + hidden_half_idx * 4 * 2)
    w_1 = tl.load(weights + hidden_half_idx * 4 * 2 + 1)
    r_0 += w_0 * gaussian_pdf
    r_1 += w_1 * gaussian_pdf

    # Compute Gaussian component 1
    d_0 = x_0 - mean_1_0
    d_1 = x_1 - mean_1_1
    sigma_d_0 = d_0 * ivc10 + d_1 * ivc11
    sigma_d_1 = d_0 * ivc11 + d_1 * ivc10
    # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 4 + seq_idx * HIDDEN * 4 + hidden_half_idx * 2 * 4 + 2, sigma_d_0)
    # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN * 2 + hidden_half_idx * 2 * 4 + 3, sigma_d_1)
    matmul_0 = -0.5 * d_0 * sigma_d_0
    matmul_1 = -0.5 * d_1 * sigma_d_1
    gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det1) * two_pi)
    # tl.store(gaussians + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN*2 + hidden_half_idx*2 + 1, gaussian_pdf)
    w_0 = tl.load(weights + hidden_half_idx * 4 * 2 +2)
    w_1 = tl.load(weights + hidden_half_idx * 4 * 2 + 3)
    r_0 += w_0 * gaussian_pdf
    r_1 += w_1 * gaussian_pdf

    # Compute Gaussian component 2
    d_0 = x_0 - mean_2_0
    d_1 = x_1 - mean_2_1
    sigma_d_0 = d_0 * ivc20 + d_1 * ivc21
    sigma_d_1 = d_0 * ivc21 + d_1 * ivc20
    # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 4 + seq_idx * HIDDEN * 4 + hidden_half_idx * 2 * 4 + 4, sigma_d_0)
    # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN * 2 + hidden_half_idx * 2 * 4 + 5, sigma_d_1)
    matmul_0 = -0.5 * d_0 * sigma_d_0
    matmul_1 = -0.5 * d_1 * sigma_d_1
    gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det2) * two_pi)
    # tl.store(gaussians + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN*2 + hidden_half_idx*2 + 2, gaussian_pdf)
    w_0 = tl.load(weights + hidden_half_idx * 4 * 2 +4)
    w_1 = tl.load(weights + hidden_half_idx * 4 * 2 + 5)
    r_0 += w_0 * gaussian_pdf
    r_1 += w_1 * gaussian_pdf

    # Compute Gaussian component 3
    d_0 = x_0 - mean_3_0
    d_1 = x_1 - mean_3_1
    sigma_d_0 = d_0 * ivc30 + d_1 * ivc31
    sigma_d_1 = d_0 * ivc31 + d_1 * ivc30
    # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 4 + seq_idx * HIDDEN * 4 + hidden_half_idx * 2 * 4 + 6, sigma_d_0)
    # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN * 2 + hidden_half_idx * 2 * 4 + 7, sigma_d_1)
    matmul_0 = -0.5 * d_0 * sigma_d_0
    matmul_1 = -0.5 * d_1 * sigma_d_1
    gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det3) * two_pi)
    # tl.store(gaussians + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN*2 + hidden_half_idx*2 + 3, gaussian_pdf)
    w_0 = tl.load(weights + hidden_half_idx * 4 * 2 + 6)
    w_1 = tl.load(weights + hidden_half_idx * 4 * 2 + 7)
    r_0 += w_0 * gaussian_pdf
    r_1 += w_1 * gaussian_pdf

    tl.store(output + idx_base, r_0)
    tl.store(output + idx_base + 1, r_1)


@triton.jit
def gmm2d_backward(x, weights, grad_output, grad_x, grad_weights, SEQ_LEN: tl.constexpr, HIDDEN: tl.constexpr):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_half_idx = tl.program_id(2)

    pi = 3.141592653589793
    two_pi = 2 * pi

    idx_base = batch_idx * SEQ_LEN * HIDDEN + seq_idx * HIDDEN + hidden_half_idx * 2
    
    x_0 = tl.load(x + idx_base)
    x_1 = tl.load(x + idx_base + 1)

    mean_0_0, mean_0_1 = 0.0, 0.0
    mean_1_0, mean_1_1 = 0.0, 1.0
    mean_2_0, mean_2_1 = 1.0, 0.0
    mean_3_0, mean_3_1 = 1.0, 1.0

    ivc00, ivc01 = 0.4453, 0.0057
    ivc10, ivc11 = 0.4465, -0.0024
    ivc20, ivc21 = 0.4457, -0.0034
    ivc30, ivc31 = 0.4466, 0.0025

    det0, det1, det2, det3 = 5.0428, 5.0161, 5.0334, 5.0148
    
    grad_out_0 = tl.load(grad_output + idx_base)
    grad_out_1 = tl.load(grad_output + idx_base + 1)

    grad_x_0 = 0.0
    grad_x_1 = 0.0

    # Compute Gaussian component 0
    d_0 = x_0 - mean_0_0
    d_1 = x_1 - mean_0_1
    sigma_d_0 = d_0 * ivc00 + d_1 * ivc01
    sigma_d_1 = d_0 * ivc01 + d_1 * ivc00
    matmul_0 = -0.5 * d_0 * sigma_d_0
    matmul_1 = -0.5 * d_1 * sigma_d_1
    gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det0) * two_pi)
    
    w_offsets = hidden_half_idx * 4 * 2
    tl.atomic_add(grad_weights + w_offsets, grad_out_0 * gaussian_pdf)
    tl.atomic_add(grad_weights + w_offsets + 1, grad_out_1 * gaussian_pdf)
    
    w_0 = tl.load(weights + w_offsets)
    w_1 = tl.load(weights + w_offsets + 1)
    
    out_0_contrib_x0 = -gaussian_pdf * sigma_d_0 * w_0
    out_0_contrib_x1 = -gaussian_pdf * sigma_d_1 * w_0
    out_1_contrib_x0 = -gaussian_pdf * sigma_d_0 * w_1
    out_1_contrib_x1 = -gaussian_pdf * sigma_d_1 * w_1
    grad_x_0 += out_0_contrib_x0 * grad_out_0 + out_1_contrib_x0 * grad_out_1
    grad_x_1 += out_0_contrib_x1 * grad_out_0 + out_1_contrib_x1 * grad_out_1

    # Compute Gaussian component 1
    d_0 = x_0 - mean_1_0
    d_1 = x_1 - mean_1_1
    sigma_d_0 = d_0 * ivc10 + d_1 * ivc11
    sigma_d_1 = d_0 * ivc11 + d_1 * ivc10
    matmul_0 = -0.5 * d_0 * sigma_d_0
    matmul_1 = -0.5 * d_1 * sigma_d_1
    gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det1) * two_pi)
    
    w_offsets = hidden_half_idx * 4 * 2 + 2 
    tl.atomic_add(grad_weights + w_offsets, grad_out_0 * gaussian_pdf)
    tl.atomic_add(grad_weights + w_offsets + 1, grad_out_1 * gaussian_pdf)
    w_0 = tl.load(weights + w_offsets)
    w_1 = tl.load(weights + w_offsets + 1)
    
    out_0_contrib_x0 = -gaussian_pdf * sigma_d_0 * w_0
    out_0_contrib_x1 = -gaussian_pdf * sigma_d_1 * w_0
    out_1_contrib_x0 = -gaussian_pdf * sigma_d_0 * w_1
    out_1_contrib_x1 = -gaussian_pdf * sigma_d_1 * w_1
    grad_x_0 += out_0_contrib_x0 * grad_out_0 + out_1_contrib_x0 * grad_out_1
    grad_x_1 += out_0_contrib_x1 * grad_out_0 + out_1_contrib_x1 * grad_out_1

    # Compute Gaussian component 2
    d_0 = x_0 - mean_2_0
    d_1 = x_1 - mean_2_1
    sigma_d_0 = d_0 * ivc20 + d_1 * ivc21
    sigma_d_1 = d_0 * ivc21 + d_1 * ivc20
    matmul_0 = -0.5 * d_0 * sigma_d_0
    matmul_1 = -0.5 * d_1 * sigma_d_1
    gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det2) * two_pi)

    w_offsets = hidden_half_idx * 4 * 2 + 4
    tl.atomic_add(grad_weights + w_offsets, grad_out_0 * gaussian_pdf)
    tl.atomic_add(grad_weights + w_offsets + 1, grad_out_1 * gaussian_pdf)
  
    w_0 = tl.load(weights + w_offsets)
    w_1 = tl.load(weights + w_offsets + 1)
    
    out_0_contrib_x0 = -gaussian_pdf * sigma_d_0 * w_0
    out_0_contrib_x1 = -gaussian_pdf * sigma_d_1 * w_0
    out_1_contrib_x0 = -gaussian_pdf * sigma_d_0 * w_1
    out_1_contrib_x1 = -gaussian_pdf * sigma_d_1 * w_1
    grad_x_0 += out_0_contrib_x0 * grad_out_0 + out_1_contrib_x0 * grad_out_1
    grad_x_1 += out_0_contrib_x1 * grad_out_0 + out_1_contrib_x1 * grad_out_1

    # Compute Gaussian component 3
    d_0 = x_0 - mean_3_0
    d_1 = x_1 - mean_3_1
    sigma_d_0 = d_0 * ivc30 + d_1 * ivc31
    sigma_d_1 = d_0 * ivc31 + d_1 * ivc30
    matmul_0 = -0.5 * d_0 * sigma_d_0
    matmul_1 = -0.5 * d_1 * sigma_d_1
    gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det3) * two_pi)

    w_offsets = hidden_half_idx * 4 * 2 + 6
    tl.atomic_add(grad_weights + w_offsets, grad_out_0 * gaussian_pdf)
    tl.atomic_add(grad_weights + w_offsets + 1, grad_out_1 * gaussian_pdf)
   
    w_0 = tl.load(weights + w_offsets)
    w_1 = tl.load(weights + w_offsets + 1)

    out_0_contrib_x0 = -gaussian_pdf * sigma_d_0 * w_0
    out_0_contrib_x1 = -gaussian_pdf * sigma_d_1 * w_0
    out_1_contrib_x0 = -gaussian_pdf * sigma_d_0 * w_1
    out_1_contrib_x1 = -gaussian_pdf * sigma_d_1 * w_1
    grad_x_0 += out_0_contrib_x0 * grad_out_0 + out_1_contrib_x0 * grad_out_1
    grad_x_1 += out_0_contrib_x1 * grad_out_0 + out_1_contrib_x1 * grad_out_1

    tl.store(grad_x + idx_base, grad_x_0)
    tl.store(grad_x + idx_base + 1, grad_x_1)


class GMM2DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights):
        batch_size, seq_len, hidden = x.shape
        ctx.save_for_backward(x, weights)
        
        output = torch.empty_like(x)
        grid = (batch_size, seq_len, hidden // 2)
        gmm2d[grid](x, weights, output, seq_len, hidden)
        ctx.save_for_backward(x, weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weights = ctx.saved_tensors
        batch_size, seq_len, hidden = x.shape
        
        grad_x = torch.zeros_like(x)

        grad_weights = torch.zeros_like(weights)

        grid = (batch_size, seq_len, hidden // 2)
        gmm2d_backward[grid](x, weights, grad_output, grad_x, grad_weights, seq_len, hidden)

        return grad_x, grad_weights

gmm2d_autograd = GMM2DTriton.apply

import torch
import triton
import triton.language as tl


# @triton.jit
# def gmm2d(x, weights, output, SEQ_LEN: tl.constexpr, HIDDEN: tl.constexpr):
#     batch_idx = tl.program_id(0)
#     seq_idx = tl.program_id(1)
#     hidden_half_idx = tl.program_id(2)

#     pi = 3.141592653589793
#     two_pi = 2 * pi

#     idx_base = batch_idx * SEQ_LEN * HIDDEN + seq_idx * HIDDEN + hidden_half_idx * 2
    
#     x_0 = tl.load(x + idx_base)
#     x_1 = tl.load(x + idx_base + 1)

#     r_0 = 0.0
#     r_1 = 0.0

#     mean_0_0, mean_0_1 = 0.0, 0.0
#     mean_1_0, mean_1_1 = 0.0, 1.0
#     mean_2_0, mean_2_1 = 1.0, 0.0
#     mean_3_0, mean_3_1 = 1.0, 1.0

#     ivc00, ivc01 = 0.4453, 0.0057
#     ivc10, ivc11 = 0.4465, -0.0024
#     ivc20, ivc21 = 0.4457, -0.0034
#     ivc30, ivc31 = 0.4466, 0.0025

#     det0, det1, det2, det3 = 5.0428, 5.0161, 5.0334, 5.0148
    
#     # Compute Gaussian component 0
#     d_0 = x_0 - mean_0_0
#     d_1 = x_1 - mean_0_1
#     sigma_d_0 = d_0 * ivc00 + d_1 * ivc01
#     sigma_d_1 = d_0 * ivc01 + d_1 * ivc00
   
#     matmul_0 = -0.5 * d_0 * sigma_d_0
#     matmul_1 = -0.5 * d_1 * sigma_d_1
#     gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det0) * two_pi)
#     # tl.store(gaussians + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN*2 + hidden_half_idx*2, gaussian_pdf)
#     w_0 = tl.load(weights + hidden_half_idx * 4 * 2)
#     w_1 = tl.load(weights + hidden_half_idx * 4 * 2 + 1)
#     r_0 += w_0 * gaussian_pdf
#     r_1 += w_1 * gaussian_pdf

#     # Compute Gaussian component 1
#     d_0 = x_0 - mean_1_0
#     d_1 = x_1 - mean_1_1
#     sigma_d_0 = d_0 * ivc10 + d_1 * ivc11
#     sigma_d_1 = d_0 * ivc11 + d_1 * ivc10
#     # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 4 + seq_idx * HIDDEN * 4 + hidden_half_idx * 2 * 4 + 2, sigma_d_0)
#     # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN * 2 + hidden_half_idx * 2 * 4 + 3, sigma_d_1)
#     matmul_0 = -0.5 * d_0 * sigma_d_0
#     matmul_1 = -0.5 * d_1 * sigma_d_1
#     gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det1) * two_pi)
#     # tl.store(gaussians + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN*2 + hidden_half_idx*2 + 1, gaussian_pdf)
#     w_0 = tl.load(weights + hidden_half_idx * 4 * 2 +2)
#     w_1 = tl.load(weights + hidden_half_idx * 4 * 2 + 3)
#     r_0 += w_0 * gaussian_pdf
#     r_1 += w_1 * gaussian_pdf

#     # Compute Gaussian component 2
#     d_0 = x_0 - mean_2_0
#     d_1 = x_1 - mean_2_1
#     sigma_d_0 = d_0 * ivc20 + d_1 * ivc21
#     sigma_d_1 = d_0 * ivc21 + d_1 * ivc20
#     # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 4 + seq_idx * HIDDEN * 4 + hidden_half_idx * 2 * 4 + 4, sigma_d_0)
#     # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN * 2 + hidden_half_idx * 2 * 4 + 5, sigma_d_1)
#     matmul_0 = -0.5 * d_0 * sigma_d_0
#     matmul_1 = -0.5 * d_1 * sigma_d_1
#     gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det2) * two_pi)
#     # tl.store(gaussians + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN*2 + hidden_half_idx*2 + 2, gaussian_pdf)
#     w_0 = tl.load(weights + hidden_half_idx * 4 * 2 +4)
#     w_1 = tl.load(weights + hidden_half_idx * 4 * 2 + 5)
#     r_0 += w_0 * gaussian_pdf
#     r_1 += w_1 * gaussian_pdf

#     # Compute Gaussian component 3
#     d_0 = x_0 - mean_3_0
#     d_1 = x_1 - mean_3_1
#     sigma_d_0 = d_0 * ivc30 + d_1 * ivc31
#     sigma_d_1 = d_0 * ivc31 + d_1 * ivc30
#     # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 4 + seq_idx * HIDDEN * 4 + hidden_half_idx * 2 * 4 + 6, sigma_d_0)
#     # tl.store(sigma_diff_products + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN * 2 + hidden_half_idx * 2 * 4 + 7, sigma_d_1)
#     matmul_0 = -0.5 * d_0 * sigma_d_0
#     matmul_1 = -0.5 * d_1 * sigma_d_1
#     gaussian_pdf = tl.exp(matmul_0 + matmul_1) / (tl.sqrt(det3) * two_pi)
#     # tl.store(gaussians + batch_idx * SEQ_LEN * HIDDEN * 2 + seq_idx * HIDDEN*2 + hidden_half_idx*2 + 3, gaussian_pdf)
#     w_0 = tl.load(weights + hidden_half_idx * 4 * 2 + 6)
#     w_1 = tl.load(weights + hidden_half_idx * 4 * 2 + 7)
#     r_0 += w_0 * gaussian_pdf
#     r_1 += w_1 * gaussian_pdf

#     tl.store(output + idx_base, r_0)
#     tl.store(output + idx_base + 1, r_1)

@triton.jit
def gmm2d(x, means, inv_var_covar, log_weights, output, SEQ_LEN: tl.constexpr, HIDDEN: tl.constexpr, NUM_COMPONENTS: tl.constexpr):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_half_idx = tl.program_id(2)
    
    idx_base = batch_idx * SEQ_LEN * HIDDEN + seq_idx * HIDDEN + hidden_half_idx * 2
    
    x_0 = tl.load(x + idx_base)
    x_1 = tl.load(x + idx_base + 1)
    
    r_0 = 0.0
    r_1 = 0.0
    
    for comp_idx in range(NUM_COMPONENTS):
        mean_0 = tl.load(means + comp_idx * 2)
        mean_1 = tl.load(means + comp_idx * 2 + 1)
        
        ivc_00 = tl.load(inv_var_covar + comp_idx * 4)
        ivc_01 = tl.load(inv_var_covar + comp_idx * 4 + 1)
        ivc_11 = tl.load(inv_var_covar + comp_idx * 4 + 3)
        
        d_0 = x_0 - mean_0
        d_1 = x_1 - mean_1
        sigma_d_0 = d_0 * ivc_00 + d_1 * ivc_01
        sigma_d_1 = d_0 * ivc_01 + d_1 * ivc_11
        
        matmul_0 = -0.5 * d_0 * sigma_d_0
        matmul_1 = -0.5 * d_1 * sigma_d_1
        log_gaussian = matmul_0 + matmul_1
        
        w_0 = tl.load(log_weights + comp_idx * 2)
        w_1 = tl.load(log_weights + comp_idx * 2 + 1)
        
        r_0 += tl.exp(w_0 + log_gaussian)
        r_1 += tl.exp(w_1 + log_gaussian)
    
    tl.store(output + idx_base, r_0)
    tl.store(output + idx_base + 1, r_1)


@triton.jit
def gmm2d_backward(x, means, inv_var_covar, log_weights, grad_output, grad_x, grad_means, grad_inv_var_covar, grad_log_weights, SEQ_LEN: tl.constexpr, HIDDEN: tl.constexpr, NUM_COMPONENTS: tl.constexpr):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_half_idx = tl.program_id(2)
    
    idx_base = batch_idx * SEQ_LEN * HIDDEN + seq_idx * HIDDEN + hidden_half_idx * 2
    
    x_0 = tl.load(x + idx_base)
    x_1 = tl.load(x + idx_base + 1)
    grad_out_0 = tl.load(grad_output + idx_base)
    grad_out_1 = tl.load(grad_output + idx_base + 1)
    
    grad_x_0 = 0.0
    grad_x_1 = 0.0
    
    for comp_idx in range(NUM_COMPONENTS):
        mean_0 = tl.load(means + comp_idx * 2)
        mean_1 = tl.load(means + comp_idx * 2 + 1)
        
        ivc_00 = tl.load(inv_var_covar + comp_idx * 4)
        ivc_01 = tl.load(inv_var_covar + comp_idx * 4 + 1)
        ivc_11 = tl.load(inv_var_covar + comp_idx * 4 + 3)
        
        d_0 = x_0 - mean_0
        d_1 = x_1 - mean_1
        sigma_d_0 = d_0 * ivc_00 + d_1 * ivc_01
        sigma_d_1 = d_0 * ivc_01 + d_1 * ivc_11
        
        matmul_0 = -0.5 * d_0 * sigma_d_0
        matmul_1 = -0.5 * d_1 * sigma_d_1
        log_gaussian = matmul_0 + matmul_1
        gaussian = tl.exp(log_gaussian)
        
        log_w_0 = tl.load(log_weights + comp_idx * 2)
        log_w_1 = tl.load(log_weights + comp_idx * 2 + 1)
        w_0 = tl.exp(log_w_0)
        w_1 = tl.exp(log_w_1)
        
        # Compute gradients with respect to mixture coefficients (log weights)
        tl.atomic_add(grad_log_weights + comp_idx * 2, grad_out_0 * gaussian*w_0)
        tl.atomic_add(grad_log_weights + comp_idx * 2 + 1, grad_out_1 * gaussian*w_1)
        
        # Update gradients with respect to input
        out_0_contrib_x0 = -gaussian * sigma_d_0 * w_0
        out_0_contrib_x1 = -gaussian * sigma_d_1 * w_0
        out_1_contrib_x0 = -gaussian * sigma_d_0 * w_1
        out_1_contrib_x1 = -gaussian * sigma_d_1 * w_1
        
        new_x_0_contrib = out_0_contrib_x0 * grad_out_0 + out_1_contrib_x0 * grad_out_1
        new_x_1_contrib = out_0_contrib_x1 * grad_out_0 + out_1_contrib_x1 * grad_out_1
        grad_x_0 += new_x_0_contrib
        grad_x_1 += new_x_1_contrib
        
        # Update gradients with respect to means
        tl.atomic_add(grad_means + comp_idx * 2, -new_x_0_contrib)
        tl.atomic_add(grad_means + comp_idx * 2 + 1, -new_x_1_contrib)

        grad_inv_00 = -0.5 * gaussian * (w_0 * d_0 * d_0 * grad_out_0 + w_1 * d_0 * d_0 * grad_out_1)
        grad_inv_01 = -0.5 * gaussian * (w_0 * d_0 * d_1 * grad_out_0 + w_1 * d_0 * d_1 * grad_out_1)
        grad_inv_11 = -0.5 * gaussian * (w_0 * d_1 * d_1 * grad_out_0 + w_1 * d_1 * d_1 * grad_out_1)
        
        tl.atomic_add(grad_inv_var_covar + comp_idx * 4, grad_inv_00)
        tl.atomic_add(grad_inv_var_covar + comp_idx * 4 + 1, grad_inv_01)
        tl.atomic_add(grad_inv_var_covar + comp_idx * 4 + 2, grad_inv_01)
        tl.atomic_add(grad_inv_var_covar + comp_idx * 4 + 3, grad_inv_11)
    
    tl.store(grad_x + idx_base, grad_x_0)
    tl.store(grad_x + idx_base + 1, grad_x_1)


class GMM2DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, means, inv_var_covar, log_weights):
        batch_size, seq_len, hidden = x.shape
        num_components = means.shape[2]
        
        output = torch.empty_like(x)
        grid = (batch_size, seq_len, hidden // 2)
        gmm2d[grid](x, means, inv_var_covar, log_weights, output, seq_len, hidden, num_components)
        ctx.save_for_backward(x, means, inv_var_covar, log_weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, means, inv_var_covar, log_weights  = ctx.saved_tensors
        batch_size, seq_len, hidden = x.shape
        num_components = means.shape[2]
        grad_x = torch.zeros_like(x)
        grad_inv_var_covar = torch.zeros_like(inv_var_covar)

        grad_log_weights = torch.zeros_like(log_weights)
        grad_means = torch.zeros_like(means)

        grid = (batch_size, seq_len, hidden // 2)
        gmm2d_backward[grid](x, means, inv_var_covar, log_weights, grad_output, grad_x, grad_means, grad_inv_var_covar, grad_log_weights, seq_len, hidden, num_components)

        return grad_x, grad_means, grad_inv_var_covar, grad_log_weights

gmm2d_autograd = GMM2DTriton.apply

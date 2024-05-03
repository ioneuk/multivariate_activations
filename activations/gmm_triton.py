import torch
import triton
import triton.language as tl


@triton.jit
def gmm2d(x, means, inv_var_covar, det_var_covar, weights, output, BATCH: tl.constexpr, SEQ_LEN: tl.constexpr, HIDDEN_HALF: tl.constexpr):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_half_idx = tl.program_id(2)

    pi = 3.141592653589793
    two_pi = 2 * pi

    idx_base = batch_idx * SEQ_LEN * HIDDEN_HALF * 2 + seq_idx * HIDDEN_HALF * 2 + hidden_half_idx * 2
    
    x_0 = tl.load(x + idx_base)
    x_1 = tl.load(x + idx_base + 1)

    # r_0 = tl.zeros((1,), dtype=tl.float32)
    # r_1 = tl.zeros((1,), dtype=tl.float32)
    r_0 = 0.0
    r_1 = 0.0

    for i in range(4):
        mean_0 = tl.load(means + i * 2)
        mean_1 = tl.load(means + i * 2 + 1)
        inv_cov_mat_0 = tl.load(inv_var_covar + i * 4)
        inv_cov_mat_1 = tl.load(inv_var_covar + i * 4+1)
        
        d_0 = x_0 - mean_0
        d_1 = x_1 - mean_1
       
        matmul_0 = -0.5 * d_0 * (d_0* inv_cov_mat_0 + d_1*inv_cov_mat_1)
        matmul_1 = -0.5* d_1 * (d_0* inv_cov_mat_1 + d_1*inv_cov_mat_0)

        det = tl.load(det_var_covar + i)
        normalization = tl.sqrt(det) * two_pi

        gaussian_pdf = tl.exp(matmul_0+matmul_1) / normalization
        w_offsets = hidden_half_idx * 4 * 2 + i * 2
        w_0 = tl.load(weights + w_offsets)
        w_1 = tl.load(weights + w_offsets + 1)
        
        r_0 += w_0 * gaussian_pdf
        r_1 += w_1 * gaussian_pdf

    tl.store(output + idx_base, r_0)
    tl.store(output + idx_base + 1, r_1)


@triton.jit
def gmm2d_backward(x, means, inv_var_covar, det_var_covar, weights, grad_output, grad_x, grad_weights, BATCH: tl.constexpr, SEQ_LEN: tl.constexpr, HIDDEN_HALF: tl.constexpr):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_half_idx = tl.program_id(2)

    pi = 3.141592653589793
    two_pi = 2 * pi

    idx_base = batch_idx * SEQ_LEN * HIDDEN_HALF * 2 + seq_idx * HIDDEN_HALF * 2 + hidden_half_idx * 2
    
    x_0 = tl.load(x + idx_base)
    x_1 = tl.load(x + idx_base + 1)
    
    grad_out_0 = tl.load(grad_output + idx_base)
    grad_out_1 = tl.load(grad_output + idx_base + 1)

    grad_x_0 = 0.0
    grad_x_1 = 0.0

    for i in range(4):
        mean_0 = tl.load(means + i * 2)
        mean_1 = tl.load(means + i * 2 + 1)
        inv_cov_mat_0 = tl.load(inv_var_covar + i * 4)
        inv_cov_mat_1 = tl.load(inv_var_covar + i * 4 + 1)
        
        d_0 = x_0 - mean_0
        d_1 = x_1 - mean_1
       
        matmul_0 = -0.5 * d_0 * (d_0* inv_cov_mat_0 + d_1*inv_cov_mat_1)
        matmul_1 = -0.5 * d_1 * (d_0* inv_cov_mat_1 + d_1*inv_cov_mat_0)

        det = tl.load(det_var_covar + i)
        normalization = tl.sqrt(det) * two_pi

        gaussian_pdf = tl.exp(matmul_0+matmul_1) / normalization

        # Update gradients with respect to weights
        w_offsets = hidden_half_idx * 4 * 2 + i * 2
        tl.atomic_add(grad_weights + w_offsets, grad_out_0 * gaussian_pdf)
        tl.atomic_add(grad_weights + w_offsets + 1, grad_out_1 * gaussian_pdf)

        # Calculate gradient with respect to inputs
        w_0 = tl.load(weights + w_offsets)
        w_1 = tl.load(weights + w_offsets + 1)
        
        grad_contrib_x0 = gaussian_pdf * (d_0 * inv_cov_mat_0 + d_1 * inv_cov_mat_1) * w_0
        grad_contrib_x1 = gaussian_pdf * (d_0 * inv_cov_mat_1 + d_1 * inv_cov_mat_0) * w_1

        grad_x_0 -= grad_contrib_x0 * grad_out_0
        grad_x_1 -= grad_contrib_x1 * grad_out_1

    # Store calculated gradients back to gradient tensors
    tl.store(grad_x + idx_base, grad_x_0)
    tl.store(grad_x + idx_base + 1, grad_x_1)



class GMM2DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, means, inv_var_covar, det_var_covar, weights):
        batch_size, seq_len, hidden = x.shape
        x = x.view(batch_size, seq_len, hidden // 2, 2)
        ctx.save_for_backward(x, means, inv_var_covar, det_var_covar, weights)
        
        output = torch.empty_like(x)

        grid = (batch_size, seq_len, hidden // 2)
        gmm2d[grid](x, means, inv_var_covar, det_var_covar, weights, output, batch_size, seq_len, hidden // 2)

        return output.reshape(batch_size, seq_len, hidden)

    @staticmethod
    def backward(ctx, grad_output):
        x, means, inv_var_covar, det_var_covar, weights = ctx.saved_tensors
        batch_size, seq_len, hidden_half, _ = x.shape
        
        grad_x = torch.zeros_like(x)

        grad_weights = torch.zeros_like(weights)

        grad_output = grad_output.view(batch_size, seq_len, hidden_half, 2)
        grid = (batch_size, seq_len, hidden_half)
        gmm2d_backward[grid](x, means, inv_var_covar, det_var_covar, weights, grad_output, grad_x, grad_weights, batch_size, seq_len, hidden_half)

        return grad_x.reshape(batch_size, seq_len, -1), None, None, None, grad_weights

gmm2d_autograd = GMM2DTriton.apply

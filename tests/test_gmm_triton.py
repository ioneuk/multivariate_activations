import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from activations.gmm2d import GMMActivation2D, gmm2d_precomputed

@pytest.fixture
def setup_gmm():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32  # Use double precision for gradcheck accuracy

    # Define dimensions
    batch_size = 10
    seq_len = 20
    hidden = 30  # Ensure this is even for the test

    
    # Prepare the GMMActivation2D module
    gmm_activation = GMMActivation2D(dim=hidden, device=device, dtype=dtype, use_triton=True)

    x = torch.randn(batch_size, seq_len, hidden, dtype=dtype, device=device, requires_grad=True)
    means = gmm_activation.modes
    inv_var_covar = gmm_activation.inv_var_covar
    det = gmm_activation.det_var_covar
    weights = gmm_activation.weights

    return gmm_activation, x, means, inv_var_covar, det, weights

def test_output_shape(setup_gmm):
    gmm_activation, x, _, _, _, _ = setup_gmm
    output = gmm_activation(x)
    assert output.shape == (x.shape[0], x.shape[1], gmm_activation.dim)

def test_correctness(setup_gmm):
    gmm_activation, x, means, inv_var_covar, det, weights = setup_gmm
    expected_output = gmm2d_precomputed(x, means, inv_var_covar, det, weights)
    output = gmm_activation(x)
    torch.testing.assert_allclose(output, expected_output, atol=1e-3, rtol=0)

def test_gradient_wrt_input_finite_diff(setup_gmm):
    gmm_activation, x, _, __, ___, ____ = setup_gmm
    params = (x,)
    assert gradcheck(gmm_activation, params, eps=1e-3, atol=1e-3, rtol=0)

def test_gradient_wrt_input_finite_diff_ref_impl(setup_gmm):
    _, x, means, inv_var_covar, det, weights = setup_gmm
    params = (x, means, inv_var_covar, det, weights)
    assert gradcheck(gmm2d_precomputed, params, eps=1e-3, atol=1e-3, rtol=0)

def test_gradient_wrt_weight_ref_impl(setup_gmm):
    gmm_activation, x, means, inv_var_covar, det, weights = setup_gmm
    x.requires_grad = True
    weights.requires_grad = True
    means.requires_grad = True
    inv_var_covar.requires_grad = True
    det.requires_grad = True
   
    expected_output = gmm2d_precomputed(x, means, inv_var_covar, det, weights)
    dl_doutput = 1*torch.randn_like(expected_output, requires_grad=True) 
    expected_output.backward(gradient=dl_doutput, inputs=weights)
    expected_grad = weights.grad.clone()
    weights.grad = None 
    x.grad = None

    actual_output = gmm_activation(x)
    torch.testing.assert_allclose(expected_output, actual_output, atol=1e-3, rtol=0)

    actual_output.backward(gradient=dl_doutput, inputs=weights)
    actual_grad = weights.grad.clone()
    torch.testing.assert_allclose(expected_grad, actual_grad, atol=1e-3, rtol=0)

def test_gradient_wrt_input_ref_impl(setup_gmm):
    gmm_activation, x, means, inv_var_covar, det, weights = setup_gmm
    x.requires_grad = True
    weights.requires_grad = True
    means.requires_grad = True
    inv_var_covar.requires_grad = True
    det.requires_grad = True
   
    expected_output = gmm2d_precomputed(x, means, inv_var_covar, det, weights)
    dl_doutput = 1*torch.randn_like(expected_output, requires_grad=True) 
    expected_output.backward(gradient=dl_doutput, inputs=x)
    expected_grad = x.grad.clone()
    weights.grad = None 
    x.grad = None

    actual_output = gmm_activation(x)
    torch.testing.assert_allclose(expected_output, actual_output, atol=1e-3, rtol=0)
    actual_output.backward(gradient=dl_doutput, inputs=x)
    actual_grad = x.grad.clone()

    print(expected_grad)
    print(actual_grad)
    torch.testing.assert_allclose(expected_grad, actual_grad, atol=1e-3, rtol=0)

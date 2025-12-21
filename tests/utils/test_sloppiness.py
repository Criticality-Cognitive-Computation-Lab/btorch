import torch

from btorch.utils.sloppiness import (
    hessian_matrix,
    hessian_vector_product,
    sloppiness_spectrum,
)


def _make_linear_problem(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2),
        torch.nn.Tanh(),
        torch.nn.Linear(2, 1),
    )
    x = torch.randn(5, 3)
    y = torch.randn(5, 1)

    def loss_fn():
        return torch.nn.functional.mse_loss(model(x), y)

    return model, loss_fn


def test_hvp_matches_explicit_hessian():
    model, loss_fn = _make_linear_problem(seed=0)
    params = list(model.parameters())
    v = torch.randn(sum(p.numel() for p in params))

    loss = loss_fn()
    hvp = hessian_vector_product(loss, params, v)
    hess = hessian_matrix(loss_fn, params)
    hvp_ref = hess @ v

    assert torch.allclose(hvp, hvp_ref, atol=1e-4, rtol=1e-4)


def test_randomized_spectrum_close_to_exact():
    model, loss_fn = _make_linear_problem(seed=1)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    k = min(5, n_params)

    exact_evals = sloppiness_spectrum(
        loss_fn, params, k=k, method="exact", return_vectors=False
    )
    rand_evals = sloppiness_spectrum(
        loss_fn,
        params,
        k=k,
        method="randomized",
        oversample=5,
        n_iter=2,
        return_vectors=False,
    )

    assert torch.allclose(rand_evals, exact_evals, atol=5e-3, rtol=5e-2)

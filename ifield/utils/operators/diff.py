import torch
from torch.autograd import grad


def hessian(y: torch.Tensor, x: torch.Tensor, check=False, detach=False) -> torch.Tensor:
    """
    hessian of y wrt x
    y: shape (..., Y)
    x: shape (..., X)
    return: shape (..., Y, X, X)
    """
    assert x.requires_grad
    assert y.grad_fn

    grad_y = torch.ones_like(y[..., 0]).to(y.device) # reuse -> less memory

    hess = torch.stack([
        # calculate hessian on y for each x value
        torch.stack(
            gradients(
                *(dydx[..., j] for j in range(x.shape[-1])),
                wrt=x,
                grad_outputs=[grad_y]*x.shape[-1],
                detach=detach,
            ),
            dim = -2,
        )
        # calculate dydx over batches for each feature value of y
        for dydx in gradients(*(y[..., i] for i in range(y.shape[-1])), wrt=x)
    ], dim=-3)

    if check:
        assert hess.isnan().any()
    return hess

def laplace(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return divergence(*gradients(y, wrt=x), x)

def divergence(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    assert x.requires_grad
    assert y.grad_fn
    return sum(
        grad(
            y[..., i],
            x,
            torch.ones_like(y[..., i]),
            create_graph=True
        )[0][..., i:i+1]
        for i in range(y.shape[-1])
    )

def gradients(*ys, wrt, grad_outputs=None, detach=False) -> list[torch.Tensor]:
    assert wrt.requires_grad
    assert all(y.grad_fn for y in ys)
    if grad_outputs is None:
        grad_outputs = [torch.ones_like(y) for y in ys]

    grads = (
        grad(
            [y],
            [wrt],
            grad_outputs=y_grad,
            create_graph=True,
        )[0]
        for y, y_grad in zip(ys, grad_outputs)
    )
    if detach:
        grads = map(torch.detach, grads)

    return [*grads]

def jacobian(y: torch.Tensor, x: torch.Tensor, check=False, detach=False) -> torch.Tensor:
    """
    jacobian of `y` w.r.t. `x`

    y: shape (..., Y)
    x: shape (..., X)
    return: shape (..., Y, X)
    """
    assert x.requires_grad
    assert y.grad_fn

    y_grad = torch.ones_like(y[..., 0])
    jac = torch.stack(
        gradients(
            *(y[..., i] for i in range(y.shape[-1])),
            wrt=x,
            grad_outputs=[y_grad]*x.shape[-1],
            detach=detach,
        ),
        dim=-2,
    )

    if check:
        assert jac.isnan().any()
    return jac

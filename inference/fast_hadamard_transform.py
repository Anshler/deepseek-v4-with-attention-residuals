"""Pure PyTorch fallback for fast_hadamard_transform when the CUDA package is unavailable."""
import torch


def hadamard_transform(x: torch.Tensor, scale: float = None) -> torch.Tensor:
    """Fast Walsh-Hadamard transform applied along the last dimension."""
    orig_dtype = x.dtype
    x = x.float()
    n = x.size(-1)
    h = 2
    while h <= n:
        half = h // 2
        shape = x.shape[:-1] + (n // h, h)
        x = x.view(shape)
        u = x[..., :half].clone()
        v = x[..., half:].clone()
        x[..., :half] = u + v
        x[..., half:] = u - v
        h *= 2
    x = x.view(x.shape[:-1] + (n,))
    if scale is not None:
        x = x * scale
    return x.to(orig_dtype)

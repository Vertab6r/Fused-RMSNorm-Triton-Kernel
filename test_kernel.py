# checks the triton kernel against plain pytorch. run with: pytest -q

import pytest
import torch
from kernel import rmsnorm


def reference(x, w, eps=1e-6):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return out.to(x.dtype)


# 3000 is on purpose: not a power of 2, so it tests the masking
@pytest.mark.parametrize("n", [1024, 3000, 4096, 8192])
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float16, 2e-3)])
def test_matches_reference(n, dtype, tol):
    torch.manual_seed(0)

    x = torch.randn(256, n, device="cuda", dtype=dtype)
    w = torch.randn(n, device="cuda", dtype=dtype)

    torch.testing.assert_close(rmsnorm(x, w), reference(x, w), atol=tol, rtol=tol)

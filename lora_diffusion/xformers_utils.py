import functools

import torch
from diffusers.models.attention import BasicTransformerBlock
from diffusers.utils.import_utils import is_xformers_available

from .lora import LoraInjectedLinear

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


@functools.cache
def test_xformers_backwards(size):
    @torch.enable_grad()
    def _grad(size):
        q = torch.randn((1, 4, size), device="cuda")
        k = torch.randn((1, 4, size), device="cuda")
        v = torch.randn((1, 4, size), device="cuda")

        q = q.detach().requires_grad_()
        k = k.detach().requires_grad_()
        v = v.detach().requires_grad_()

        out = xformers.ops.memory_efficient_attention(q, k, v)
        loss = out.sum(2).mean(0).sum()

        return torch.autograd.grad(loss, v)

    try:
        _grad(size)
        print(size, "pass")
        return True
    except Exception as e:
        print(size, "fail")
        return False


def set_use_memory_efficient_attention_xformers(
    module: torch.nn.Module, valid: bool
) -> None:
    def fn_test_dim_head(module: torch.nn.Module):
        if isinstance(module, BasicTransformerBlock):
            # dim_head isn't stored anywhere, so back-calculate
            source = module.attn1.to_v
            if isinstance(source, LoraInjectedLinear):
                source = source.linear

            dim_head = source.out_features // module.attn1.heads

            result = test_xformers_backwards(dim_head)

            # If dim_head > dim_head_max, turn xformers off
            if not result:
                module.set_use_memory_efficient_attention_xformers(False)

        for child in module.children():
            fn_test_dim_head(child)

    if not is_xformers_available() and valid:
        print("XFormers is not available. Skipping.")
        return

    module.set_use_memory_efficient_attention_xformers(valid)

    if valid:
        fn_test_dim_head(module)

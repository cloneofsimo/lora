import fire
from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn

from .lora import (
    save_all,
    _find_modules,
    LoraInjectedConv2d,
    LoraInjectedLinear,
    inject_trainable_lora,
    inject_trainable_lora_extended,
)


def _iter_lora(model):
    for module in model.modules():
        if isinstance(module, LoraInjectedConv2d) or isinstance(
            module, LoraInjectedLinear
        ):
            yield module


def overwrite_base(base_model, tuned_model, rank, clamp_quantile):
    device = base_model.device
    dtype = base_model.dtype

    for lor_base, lor_tune in zip(_iter_lora(base_model), _iter_lora(tuned_model)):

        if isinstance(lor_base, LoraInjectedLinear):
            residual = lor_tune.linear.weight.data - lor_base.linear.weight.data
            # SVD on residual
            print("Distill Linear shape ", residual.shape)
            residual = residual.float()
            U, S, Vh = torch.linalg.svd(residual)
            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)

            Vh = Vh[:rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp_quantile)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            assert lor_base.lora_up.weight.shape == U.shape
            assert lor_base.lora_down.weight.shape == Vh.shape

            lor_base.lora_up.weight.data = U.to(device=device, dtype=dtype)
            lor_base.lora_down.weight.data = Vh.to(device=device, dtype=dtype)

        if isinstance(lor_base, LoraInjectedConv2d):
            residual = lor_tune.conv.weight.data - lor_base.conv.weight.data
            print("Distill Conv shape ", residual.shape)

            residual = residual.float()
            residual = residual.flatten(start_dim=1)

            # SVD on residual
            U, S, Vh = torch.linalg.svd(residual)
            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)

            Vh = Vh[:rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp_quantile)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            # U is (out_channels, rank) with 1x1 conv. So,
            U = U.reshape(U.shape[0], U.shape[1], 1, 1)
            # V is (rank, in_channels * kernel_size1 * kernel_size2)
            # now reshape:
            Vh = Vh.reshape(
                Vh.shape[0],
                lor_base.conv.in_channels,
                lor_base.conv.kernel_size[0],
                lor_base.conv.kernel_size[1],
            )

            assert lor_base.lora_up.weight.shape == U.shape
            assert lor_base.lora_down.weight.shape == Vh.shape

            lor_base.lora_up.weight.data = U.to(device=device, dtype=dtype)
            lor_base.lora_down.weight.data = Vh.to(device=device, dtype=dtype)


def svd_distill(
    target_model: str,
    base_model: str,
    rank: int = 4,
    clamp_quantile: float = 0.99,
    device: str = "cuda:0",
    save_path: str = "svd_distill.safetensors",
):
    pipe_base = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    ).to(device)

    pipe_tuned = StableDiffusionPipeline.from_pretrained(
        target_model, torch_dtype=torch.float16
    ).to(device)

    # Inject unet
    _ = inject_trainable_lora_extended(pipe_base.unet, r=rank)
    _ = inject_trainable_lora_extended(pipe_tuned.unet, r=rank)

    overwrite_base(
        pipe_base.unet, pipe_tuned.unet, rank=rank, clamp_quantile=clamp_quantile
    )

    # Inject text encoder
    _ = inject_trainable_lora(
        pipe_base.text_encoder, r=rank, target_replace_module={"CLIPAttention"}
    )
    _ = inject_trainable_lora(
        pipe_tuned.text_encoder, r=rank, target_replace_module={"CLIPAttention"}
    )

    overwrite_base(
        pipe_base.text_encoder,
        pipe_tuned.text_encoder,
        rank=rank,
        clamp_quantile=clamp_quantile,
    )

    save_all(
        unet=pipe_base.unet,
        text_encoder=pipe_base.text_encoder,
        placeholder_token_ids=None,
        placeholder_tokens=None,
        save_path=save_path,
        save_lora=True,
        save_ti=False,
    )


def main():
    fire.Fire(svd_distill)

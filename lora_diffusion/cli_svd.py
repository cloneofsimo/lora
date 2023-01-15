import fire
from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn

from .lora import save_all, _find_modules


def _text_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def _ti_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["ti", "pt"])


def extract_linear_weights(model, target_replace_module):
    lins = []
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        lins.append(_child_module.weight)

    return lins


def svd_distill(
    target_model: str,
    base_model: str,
    rank: int = 4,
    clamp_quantile: float = 0.99,
    device: str = "cuda:0",
    save_path: str = "svd_distill.pt",
):
    pipe_base = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    ).to(device)

    pipe_tuned = StableDiffusionPipeline.from_pretrained(
        target_model, torch_dtype=torch.float16
    ).to(device)

    ori_unet = extract_linear_weights(
        pipe_base.unet, ["CrossAttention", "Attention", "GEGLU"]
    )
    ori_clip = extract_linear_weights(pipe_base.text_encoder, ["CLIPAttention"])

    tuned_unet = extract_linear_weights(
        pipe_tuned.unet, ["CrossAttention", "Attention", "GEGLU"]
    )
    tuned_clip = extract_linear_weights(pipe_tuned.text_encoder, ["CLIPAttention"])

    diffs_unet = []
    diffs_clip = []

    for ori, tuned in zip(ori_unet, tuned_unet):
        diffs_unet.append(tuned - ori)

    for ori, tuned in zip(ori_clip, tuned_clip):
        diffs_clip.append(tuned - ori)

    uds_unet = []
    uds_clip = []
    with torch.no_grad():
        for mat in diffs_unet:
            mat = mat.float()

            U, S, Vh = torch.linalg.svd(mat)

            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)

            Vh = Vh[:rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp_quantile)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            uds_unet.append(U)
            uds_unet.append(Vh)

        for mat in diffs_clip:
            mat = mat.float()

            U, S, Vh = torch.linalg.svd(mat)

            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)

            Vh = Vh[:rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp_quantile)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            uds_clip.append(U)
            uds_clip.append(Vh)

    torch.save(uds_unet, save_path)
    torch.save(uds_clip, _text_lora_path(save_path))


def main():
    fire.Fire(svd_distill)

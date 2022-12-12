from typing import Literal, Union, Dict
import os
import shutil
import fire
from diffusers import StableDiffusionPipeline

import torch
from .lora import tune_lora_scale, weight_apply_lora
from .to_ckpt_v2 import convert_to_ckpt


def add(
    path_1: str,
    path_2: str,
    output_path: str,
    alpha: float = 0.5,
    mode: Literal["lpl", "upl", "upl-ckpt-v2"] = "lpl",
):
    if mode == "lpl":
        out_list = []
        l1 = torch.load(path_1)
        l2 = torch.load(path_2)

        l1pairs = zip(l1[::2], l1[1::2])
        l2pairs = zip(l2[::2], l2[1::2])

        for (x1, y1), (x2, y2) in zip(l1pairs, l2pairs):
            x1.data = alpha * x1.data + (1 - alpha) * x2.data
            y1.data = alpha * y1.data + (1 - alpha) * y2.data

            out_list.append(x1)
            out_list.append(y1)

        torch.save(out_list, output_path)

    elif mode == "upl":

        loaded_pipeline = StableDiffusionPipeline.from_pretrained(
            path_1,
        ).to("cpu")

        weight_apply_lora(loaded_pipeline.unet, torch.load(path_2), alpha=alpha)

        loaded_pipeline.save_pretrained(output_path)

    elif mode == "upl-ckpt-v2":

        loaded_pipeline = StableDiffusionPipeline.from_pretrained(
            path_1,
        ).to("cpu")

        weight_apply_lora(loaded_pipeline.unet, torch.load(path_2), alpha=alpha)

        _tmp_output = output_path + ".tmp"

        loaded_pipeline.save_pretrained(_tmp_output)
        convert_to_ckpt(_tmp_output, output_path, as_half=True)
        # remove the tmp_output folder
        shutil.rmtree(_tmp_output)

    else:
        raise ValueError(f"Unknown mode {mode}")


def main():
    fire.Fire(add)

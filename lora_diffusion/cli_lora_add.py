from typing import Literal, Union, Dict
import os
import shutil
import fire
from diffusers import StableDiffusionPipeline

import torch
from .lora import tune_lora_scale, weight_apply_lora
from .to_ckpt_v2 import convert_to_ckpt


def _text_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def add(
    path_1: str,
    path_2: str,
    output_path: str,
    alpha: float = 0.5,
    mode: Literal[
        "lpl",
        "upl",
        "upl-ckpt-v2",
    ] = "lpl",
    with_text_lora: bool = False,
):
    print("Lora Add, mode " + mode)
    if mode == "lpl":
        for _path_1, _path_2, opt in [(path_1, path_2, "unet")] + (
            [(_text_lora_path(path_1), _text_lora_path(path_2), "text_encoder")]
            if with_text_lora
            else []
        ):
            print("Loading", _path_1, _path_2)
            out_list = []
            if opt == "text_encoder":
                if not os.path.exists(_path_1):
                    print(f"No text encoder found in {_path_1}, skipping...")
                    continue
                if not os.path.exists(_path_2):
                    print(f"No text encoder found in {_path_1}, skipping...")
                    continue

            l1 = torch.load(_path_1)
            l2 = torch.load(_path_2)

            l1pairs = zip(l1[::2], l1[1::2])
            l2pairs = zip(l2[::2], l2[1::2])

            for (x1, y1), (x2, y2) in zip(l1pairs, l2pairs):
                # print("Merging", x1.shape, y1.shape, x2.shape, y2.shape)
                x1.data = alpha * x1.data + (1 - alpha) * x2.data
                y1.data = alpha * y1.data + (1 - alpha) * y2.data

                out_list.append(x1)
                out_list.append(y1)

            if opt == "unet":

                print("Saving merged UNET to", output_path)
                torch.save(out_list, output_path)

            elif opt == "text_encoder":
                print("Saving merged text encoder to", _text_lora_path(output_path))
                torch.save(
                    out_list,
                    _text_lora_path(output_path),
                )

    elif mode == "upl":

        loaded_pipeline = StableDiffusionPipeline.from_pretrained(
            path_1,
        ).to("cpu")

        weight_apply_lora(loaded_pipeline.unet, torch.load(path_2), alpha=alpha)
        if with_text_lora:

            weight_apply_lora(
                loaded_pipeline.text_encoder,
                torch.load(_text_lora_path(path_2)),
                alpha=alpha,
                target_replace_module=["CLIPAttention"],
            )

        loaded_pipeline.save_pretrained(output_path)

    elif mode == "upl-ckpt-v2":

        loaded_pipeline = StableDiffusionPipeline.from_pretrained(
            path_1,
        ).to("cpu")

        weight_apply_lora(loaded_pipeline.unet, torch.load(path_2), alpha=alpha)
        if with_text_lora:
            weight_apply_lora(
                loaded_pipeline.text_encoder,
                torch.load(_text_lora_path(path_2)),
                alpha=alpha,
                target_replace_module=["CLIPAttention"],
            )

        _tmp_output = output_path + ".tmp"

        loaded_pipeline.save_pretrained(_tmp_output)
        convert_to_ckpt(_tmp_output, output_path, as_half=True)
        # remove the tmp_output folder
        shutil.rmtree(_tmp_output)

    else:
        print("Unknown mode", mode)
        raise ValueError(f"Unknown mode {mode}")


def main():
    fire.Fire(add)

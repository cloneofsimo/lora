from typing import Literal, Union, Dict
import os
import shutil
import fire
from diffusers import StableDiffusionPipeline
from safetensors.torch import safe_open, save_file

import torch

try:
    from .lora import (
        tune_lora_scale,
        patch_pipe,
        collapse_lora,
        monkeypatch_remove_lora,
    )

    from .lora_manager import lora_join
    from .to_ckpt_v2 import convert_to_ckpt

except:  # allows running the repo without installing it (can mess up existing dependencies)
    from lora_diffusion import (
        tune_lora_scale,
        patch_pipe,
        collapse_lora,
        monkeypatch_remove_lora,
    )

    from lora_diffusion.lora_manager import lora_join
    from lora_diffusion.to_ckpt_v2 import convert_to_ckpt


def _text_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def add(
    path_1: str,
    path_2: str,
    output_path: str,
    alpha_1: float = 0.5,
    alpha_2: float = 0.5,
    mode: Literal[
        "lpl",
        "upl",
        "upl-ckpt-v2",
    ] = "lpl",
    with_text_lora: bool = False,
):
    print("Lora Add, mode " + mode)
    if mode == "lpl":
        if path_1.endswith(".pt") and path_2.endswith(".pt"):
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
                    x1.data = alpha_1 * x1.data + alpha_2 * x2.data
                    y1.data = alpha_1 * y1.data + alpha_2 * y2.data

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

        elif path_1.endswith(".safetensors") and path_2.endswith(".safetensors"):
            safeloras_1 = safe_open(path_1, framework="pt", device="cpu")
            safeloras_2 = safe_open(path_2, framework="pt", device="cpu")

            metadata = dict(safeloras_1.metadata())
            metadata.update(dict(safeloras_2.metadata()))

            ret_tensor = {}

            for keys in set(list(safeloras_1.keys()) + list(safeloras_2.keys())):
                if keys.startswith("text_encoder") or keys.startswith("unet"):

                    tens1 = safeloras_1.get_tensor(keys)
                    tens2 = safeloras_2.get_tensor(keys)

                    tens = alpha_1 * tens1 + alpha_2 * tens2
                    ret_tensor[keys] = tens
                else:
                    if keys in safeloras_1.keys():

                        tens1 = safeloras_1.get_tensor(keys)
                    else:
                        tens1 = safeloras_2.get_tensor(keys)

                    ret_tensor[keys] = tens1

            save_file(ret_tensor, output_path, metadata)

    elif mode == "upl":

        print(
            f"Merging UNET/CLIP from {path_1} with LoRA from {path_2} to {output_path}. Merging ratio : {alpha_1}."
        )

        loaded_pipeline = StableDiffusionPipeline.from_pretrained(
            path_1,
        ).to("cpu")

        patch_pipe(loaded_pipeline, path_2)

        collapse_lora(loaded_pipeline.unet, alpha_1)
        collapse_lora(loaded_pipeline.text_encoder, alpha_1)

        monkeypatch_remove_lora(loaded_pipeline.unet)
        monkeypatch_remove_lora(loaded_pipeline.text_encoder)

        loaded_pipeline.save_pretrained(output_path)

    elif mode == "upl-ckpt-v2":

        assert output_path.endswith(".ckpt"), "Only .ckpt files are supported"
        name = os.path.basename(output_path)[0:-5]

        print(
            f"You will be using {name} as the token in A1111 webui. Make sure {name} is unique enough token."
        )

        loaded_pipeline = StableDiffusionPipeline.from_pretrained(
            path_1,
        ).to("cpu")

        tok_dict = patch_pipe(loaded_pipeline, path_2, patch_ti=False)

        collapse_lora(loaded_pipeline.unet, alpha_1)
        collapse_lora(loaded_pipeline.text_encoder, alpha_1)

        monkeypatch_remove_lora(loaded_pipeline.unet)
        monkeypatch_remove_lora(loaded_pipeline.text_encoder)

        _tmp_output = output_path + ".tmp"

        loaded_pipeline.save_pretrained(_tmp_output)
        convert_to_ckpt(_tmp_output, output_path, as_half=True)
        # remove the tmp_output folder
        shutil.rmtree(_tmp_output)

        keys = sorted(tok_dict.keys())
        tok_catted = torch.stack([tok_dict[k] for k in keys])
        ret = {
            "string_to_token": {"*": torch.tensor(265)},
            "string_to_param": {"*": tok_catted},
            "name": name,
        }

        torch.save(ret, output_path[:-5] + ".pt")
        print(
            f"Textual embedding saved as {output_path[:-5]}.pt, put it in the embedding folder and use it as {name} in A1111 repo, "
        )
    elif mode == "ljl":
        print("Using Join mode : alpha will not have an effect here.")
        assert path_1.endswith(".safetensors") and path_2.endswith(
            ".safetensors"
        ), "Only .safetensors files are supported"

        safeloras_1 = safe_open(path_1, framework="pt", device="cpu")
        safeloras_2 = safe_open(path_2, framework="pt", device="cpu")

        total_tensor, total_metadata, _, _ = lora_join([safeloras_1, safeloras_2])
        save_file(total_tensor, output_path, total_metadata)

    else:
        print("Unknown mode", mode)
        raise ValueError(f"Unknown mode {mode}")


def main():
    fire.Fire(add)


if __name__ == "__main__":
    main()
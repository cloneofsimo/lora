import os

import fire
import torch
from lora_diffusion import (
    DEFAULT_TARGET_REPLACE,
    TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    UNET_DEFAULT_TARGET_REPLACE,
    convert_loras_to_safeloras_with_embeds,
    safetensors_available,
)

_target_by_name = {
    "unet": UNET_DEFAULT_TARGET_REPLACE,
    "text_encoder": TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
}


def convert(*paths, outpath, overwrite=False, **settings):
    """
    Converts one or more pytorch Lora and/or Textual Embedding pytorch files
    into a safetensor file.

    Pass all the input paths as arguments. Whether they are Textual Embedding
    or Lora models will be auto-detected.

    For Lora models, their name will be taken from the path, i.e.
        "lora_weight.pt" => unet
        "lora_weight.text_encoder.pt" => text_encoder

    You can also set target_modules and/or rank by providing an argument prefixed
    by the name.

    So a complete example might be something like:

    ```
    python -m lora_diffusion.cli_pt_to_safetensors lora_weight.* --outpath lora_weight.safetensor --unet.rank 8
    ```
    """
    modelmap = {}
    embeds = {}

    if os.path.exists(outpath) and not overwrite:
        raise ValueError(
            f"Output path {outpath} already exists, and overwrite is not True"
        )

    for path in paths:
        data = torch.load(path)

        if isinstance(data, dict):
            print(f"Loading textual inversion embeds {data.keys()} from {path}")
            embeds.update(data)

        else:
            name_parts = os.path.split(path)[1].split(".")
            name = name_parts[-2] if len(name_parts) > 2 else "unet"

            model_settings = {
                "target_modules": _target_by_name.get(name, DEFAULT_TARGET_REPLACE),
                "rank": 4,
            }

            prefix = f"{name}."
            
            arg_settings = { k[len(prefix) :]: v for k, v in settings.items() if k.startswith(prefix) }
            model_settings = { **model_settings, **arg_settings }

            print(f"Loading Lora for {name} from {path} with settings {model_settings}")

            modelmap[name] = (
                path,
                model_settings["target_modules"],
                model_settings["rank"],
            )

    convert_loras_to_safeloras_with_embeds(modelmap, embeds, outpath)


def main():
    fire.Fire(convert)


if __name__ == "__main__":
    main()

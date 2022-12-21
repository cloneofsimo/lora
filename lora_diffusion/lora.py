import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F

import torch.nn as nn


class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )

        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = 1.0

        nn.init.normal_(self.lora_down.weight, std=1 / r**2)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.linear(input) + self.lora_up(self.lora_down(input)) * self.scale


def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: List[str] = ["CrossAttention", "Attention", "GEGLU"],
    r: int = 4,
    loras=None,  # path to lora .pt
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = LoraInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                    )
                    _tmp.linear.weight = weight
                    if bias is not None:
                        _tmp.linear.bias = bias

                    # switch the module
                    _module._modules[name] = _tmp

                    require_grad_params.append(
                        _module._modules[name].lora_up.parameters()
                    )
                    require_grad_params.append(
                        _module._modules[name].lora_down.parameters()
                    )

                    if loras != None:
                        _module._modules[name].lora_up.weight = loras.pop(0)
                        _module._modules[name].lora_down.weight = loras.pop(0)

                    _module._modules[name].lora_up.weight.requires_grad = True
                    _module._modules[name].lora_down.weight.requires_grad = True
                    names.append(name)
    return require_grad_params, names


def extract_lora_ups_down(
    model, target_replace_module=["CrossAttention", "Attention", "GEGLU"]
):

    loras = []

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for _child_module in _module.modules():
                if _child_module.__class__.__name__ == "LoraInjectedLinear":
                    loras.append((_child_module.lora_up, _child_module.lora_down))
    if len(loras) == 0:
        raise ValueError("No lora injected.")
    return loras


def save_lora_weight(
    model,
    path="./lora.pt",
    target_replace_module=["CrossAttention", "Attention", "GEGLU"],
):
    weights = []
    for _up, _down in extract_lora_ups_down(
        model, target_replace_module=target_replace_module
    ):
        weights.append(_up.weight)
        weights.append(_down.weight)

    torch.save(weights, path)


def save_lora_as_json(model, path="./lora.json"):
    weights = []
    for _up, _down in extract_lora_ups_down(model):
        weights.append(_up.weight.detach().cpu().numpy().tolist())
        weights.append(_down.weight.detach().cpu().numpy().tolist())

    import json

    with open(path, "w") as f:
        json.dump(weights, f)


def weight_apply_lora(
    model,
    loras,
    target_replace_module=["CrossAttention", "Attention", "GEGLU"],
    alpha=1.0,
):

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for _child_module in _module.modules():
                if _child_module.__class__.__name__ == "Linear":

                    weight = _child_module.weight

                    up_weight = loras.pop(0).detach().to(weight.device)
                    down_weight = loras.pop(0).detach().to(weight.device)

                    # W <- W + U * D
                    weight = weight + alpha * (up_weight @ down_weight).type(
                        weight.dtype
                    )
                    _child_module.weight = nn.Parameter(weight)


def monkeypatch_lora(
    model,
    loras,
    target_replace_module=["CrossAttention", "Attention", "GEGLU"],
    r: int = 4,
):
    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = LoraInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r=r,
                    )
                    _tmp.linear.weight = weight

                    if bias is not None:
                        _tmp.linear.bias = bias

                    # switch the module
                    _module._modules[name] = _tmp

                    up_weight = loras.pop(0)
                    down_weight = loras.pop(0)

                    _module._modules[name].lora_up.weight = nn.Parameter(
                        up_weight.type(weight.dtype)
                    )
                    _module._modules[name].lora_down.weight = nn.Parameter(
                        down_weight.type(weight.dtype)
                    )

                    _module._modules[name].to(weight.device)


def monkeypatch_replace_lora(
    model,
    loras,
    target_replace_module=["CrossAttention", "Attention", "GEGLU"],
    r: int = 4,
):
    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "LoraInjectedLinear":

                    weight = _child_module.linear.weight
                    bias = _child_module.linear.bias
                    _tmp = LoraInjectedLinear(
                        _child_module.linear.in_features,
                        _child_module.linear.out_features,
                        _child_module.linear.bias is not None,
                        r=r,
                    )
                    _tmp.linear.weight = weight

                    if bias is not None:
                        _tmp.linear.bias = bias

                    # switch the module
                    _module._modules[name] = _tmp

                    up_weight = loras.pop(0)
                    down_weight = loras.pop(0)

                    _module._modules[name].lora_up.weight = nn.Parameter(
                        up_weight.type(weight.dtype)
                    )
                    _module._modules[name].lora_down.weight = nn.Parameter(
                        down_weight.type(weight.dtype)
                    )

                    _module._modules[name].to(weight.device)


def monkeypatch_add_lora(
    model,
    loras,
    target_replace_module=["CrossAttention", "Attention", "GEGLU"],
    alpha: float = 1.0,
    beta: float = 1.0,
):
    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "LoraInjectedLinear":

                    weight = _child_module.linear.weight

                    up_weight = loras.pop(0)
                    down_weight = loras.pop(0)

                    _module._modules[name].lora_up.weight = nn.Parameter(
                        up_weight.type(weight.dtype).to(weight.device) * alpha
                        + _module._modules[name].lora_up.weight.to(weight.device) * beta
                    )
                    _module._modules[name].lora_down.weight = nn.Parameter(
                        down_weight.type(weight.dtype).to(weight.device) * alpha
                        + _module._modules[name].lora_down.weight.to(weight.device)
                        * beta
                    )

                    _module._modules[name].to(weight.device)


def tune_lora_scale(model, alpha: float = 1.0):
    for _module in model.modules():
        if _module.__class__.__name__ == "LoraInjectedLinear":
            _module.scale = alpha


def _text_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def _ti_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["ti", "pt"])


def load_learned_embed_in_clip(
    learned_embeds_path, text_encoder, tokenizer, token=None, idempotent=False
):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    i = 1
    if not idempotent:
        while num_added_tokens == 0:
            print(f"The tokenizer already contains the token {token}.")
            token = f"{token[:-1]}-{i}>"
            print(f"Attempting to add the token {token}.")
            num_added_tokens = tokenizer.add_tokens(token)
            i += 1
    elif num_added_tokens == 0 and idempotent:
        print(f"The tokenizer already contains the token {token}.")
        print(f"Replacing {token} embedding.")

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token


def patch_pipe(
    pipe,
    unet_path,
    token: str,
    r: int = 4,
    patch_unet=True,
    patch_text=False,
    patch_ti=False,
    idempotent_token=True,
):
    assert (
        len(token) > 0
    ), "Token cannot be empty. Input token non-empty token like <s>."

    ti_path = _ti_lora_path(unet_path)
    text_path = _text_lora_path(unet_path)

    unet_has_lora = False
    text_encoder_has_lora = False

    for _module in pipe.unet.modules():
        if _module.__class__.__name__ == "LoraInjectedLinear":
            unet_has_lora = True

    for _module in pipe.text_encoder.modules():
        if _module.__class__.__name__ == "LoraInjectedLinear":
            text_encoder_has_lora = True
    if patch_unet:
        print("LoRA : Patching Unet")

        if not unet_has_lora:
            monkeypatch_lora(pipe.unet, torch.load(unet_path), r=r)
        else:
            monkeypatch_replace_lora(pipe.unet, torch.load(unet_path), r=r)

    if patch_text:
        print("LoRA : Patching text encoder")
        if not text_encoder_has_lora:
            monkeypatch_lora(
                pipe.text_encoder,
                torch.load(text_path),
                target_replace_module=["CLIPAttention"],
                r=r,
            )
        else:

            monkeypatch_replace_lora(
                pipe.text_encoder,
                torch.load(text_path),
                target_replace_module=["CLIPAttention"],
                r=r,
            )
    if patch_ti:
        print("LoRA : Patching token input")
        token = load_learned_embed_in_clip(
            ti_path,
            pipe.text_encoder,
            pipe.tokenizer,
            token,
            idempotent=idempotent_token,
        )


@torch.no_grad()
def inspect_lora(model, target_replace_module=["CrossAttention", "Attention", "GEGLU"]):

    fnorm = {k: [] for k in target_replace_module}

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "LoraInjectedLinear":
                    ups = _module._modules[name].lora_up.weight
                    downs = _module._modules[name].lora_down.weight

                    wght: torch.Tensor = downs @ ups
                    fnorm[name].append(wght.flatten().pow(2).mean().item())

    for k, v in fnorm.items():
        print(f"F norm on Current LoRA of {k} : {v}")

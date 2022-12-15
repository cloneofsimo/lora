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

        if r >= min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less than {min(in_features, out_features)}"
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
    target_replace_module: List[str] = ["CrossAttention", "Attention"],
    r: int = 4,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

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

                    _module._modules[name].lora_up.weight.requires_grad = True
                    _module._modules[name].lora_down.weight.requires_grad = True
                    names.append(name)

    return require_grad_params, names


def extract_lora_ups_down(model, target_replace_module=["CrossAttention", "Attention"]):

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
    model, path="./lora.pt", target_replace_module=["CrossAttention", "Attention"]
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
    model, loras, target_replace_module=["CrossAttention", "Attention"], alpha=1.0
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
    model, loras, target_replace_module=["CrossAttention", "Attention"]
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
    model, loras, target_replace_module=["CrossAttention", "Attention"]
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
    target_replace_module=["CrossAttention", "Attention"],
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

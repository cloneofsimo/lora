# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

import argparse
import hashlib
import inspect
import itertools
import math
import os
import random
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import fire

from lora_diffusion import (
    PivotalTuningDatasetCapation,
    extract_lora_ups_down,
    inject_trainable_lora,
    inspect_lora,
    save_lora_weight,
    save_all,
)


def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    placeholder_token,
    initializer_token,
    device="cuda:0",
):

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder=None if pretrained_vae_name_or_path else "vae",
        revision=None if pretrained_vae_name_or_path else revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_id,
    )


def text2img_dataloader(train_dataset, train_batch_size, tokenizer, vae, text_encoder):
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if examples[0].get("class_prompt_ids", None) is not None:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    return train_dataloader


@torch.autocast("cuda")
def loss_step(batch, unet, vae, text_encoder, scheduler, weight_dtype):
    latents = vae.encode(
        batch["pixel_values"].to(dtype=weight_dtype).to(unet.device)
    ).latent_dist.sample()
    latents = latents * 0.18215

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device))[0]

    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss


def train_inversion(
    unet,
    vae,
    text_encoder,
    dataloader,
    num_steps,
    scheduler,
    index_no_updates,
    optimizer,
    save_steps,
    placeholder_token_id,
    placeholder_token,
    save_path,
):

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    # Original Emb for TI
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    weight_dtype = torch.float16

    for epoch in range(math.ceil(num_steps / len(dataloader))):
        unet.eval()
        text_encoder.train()
        for batch in dataloader:

            loss = loss_step(batch, unet, vae, text_encoder, scheduler, weight_dtype)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            optimizer.zero_grad()

            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]

            global_step += 1

            if global_step % save_steps == 0:
                save_all(
                    unet=unet,
                    text_encoder=text_encoder,
                    placeholder_token_id=placeholder_token_id,
                    placeholder_token=placeholder_token,
                    save_path=os.path.join(save_path, f"step_inv_{global_step}.pt"),
                    save_lora=False,
                )

            if global_step >= num_steps:
                return


def perform_tuning(
    unet,
    vae,
    text_encoder,
    dataloader,
    num_steps,
    scheduler,
    optimizer,
    save_steps: int,
    placeholder_token_id,
    placeholder_token,
    save_path,
):

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    weight_dtype = torch.float16

    unet.train()
    text_encoder.train()

    for epoch in range(math.ceil(num_steps / len(dataloader))):
        for batch in dataloader:
            optimizer.zero_grad()

            loss = loss_step(batch, unet, vae, text_encoder, scheduler, weight_dtype)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(unet.parameters(), text_encoder.parameters()), 1.0
            )
            optimizer.step()
            progress_bar.update(1)

            global_step += 1

            if global_step % save_steps == 0:
                save_all(
                    unet,
                    text_encoder,
                    placeholder_token_id=placeholder_token_id,
                    placeholder_token=placeholder_token,
                    save_path=os.path.join(save_path, f"step_{global_step}.pt"),
                )
                moved = (
                    torch.tensor(list(itertools.chain(*inspect_lora(unet).values())))
                    .mean()
                    .item()
                )

                print("LORA Unet Moved", moved)
                moved = (
                    torch.tensor(
                        list(itertools.chain(*inspect_lora(text_encoder).values()))
                    )
                    .mean()
                    .item()
                )

                print("LORA CLIP Moved", moved)

            if global_step >= num_steps:
                return


def train(
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    train_text_encoder: bool = False,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    class_data_dir: Optional[str] = None,
    stochastic_attribute: Optional[str] = None,
    perform_inversion: bool = True,
    learnable_property: str = "object",  # not used
    placeholder_token: str = "<s>",
    initializer_token: str = "dog",
    class_prompt: Optional[str] = None,
    with_prior_preservation: bool = False,
    prior_loss_weight: float = 1.0,
    num_class_images: int = 100,
    seed: int = 42,
    resolution: int = 512,
    center_crop: bool = False,
    color_jitter: bool = True,
    train_batch_size: int = 1,
    sample_batch_size: int = 1,
    max_train_steps_tuning: int = 10000,
    max_train_steps_ti: int = 2000,
    save_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    mixed_precision="fp16",
    lora_rank: int = 4,
    lora_unet_target_modules={"CrossAttention", "Attention", "GEGLU"},
    lora_clip_target_modules={"CLIPAttention"},
    learning_rate_unet: float = 1e-5,
    learning_rate_text: float = 1e-5,
    learning_rate_ti: float = 5e-4,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 100,
    weight_decay_ti: float = 0.01,
    weight_decay_lora: float = 0.01,
    use_8bit_adam: bool = False,
    device="cuda:0",
):
    torch.manual_seed(seed)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # get the models
    text_encoder, vae, unet, tokenizer, placeholder_token_id = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_token,
        initializer_token,
        device=device,
    )

    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.gradient_checkpointing_enable()

    if scale_lr:
        unet_lr = learning_rate_unet * gradient_accumulation_steps * train_batch_size
        text_encoder_lr = (
            learning_rate_text * gradient_accumulation_steps * train_batch_size
        )
        ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size
    else:
        unet_lr = learning_rate_unet
        text_encoder_lr = learning_rate_text
        ti_lr = learning_rate_ti

    train_dataset = PivotalTuningDatasetCapation(
        instance_data_root=instance_data_dir,
        placeholder_token=placeholder_token,
        stochastic_attribute=stochastic_attribute,
        class_data_root=class_data_dir if with_prior_preservation else None,
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=resolution,
        center_crop=center_crop,
        color_jitter=color_jitter,
    )

    train_dataloader = text2img_dataloader(
        train_dataset, train_batch_size, tokenizer, vae, text_encoder
    )

    index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id

    unet.requires_grad_(False)
    vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    # STEP 1 : Perform Inversion
    if perform_inversion:
        ti_optimizer = optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),
            lr=ti_lr,
            weight_decay=weight_decay_ti,
        )

        train_inversion(
            unet,
            vae,
            text_encoder,
            train_dataloader,
            max_train_steps_ti,
            scheduler=noise_scheduler,
            index_no_updates=index_no_updates,
            optimizer=ti_optimizer,
            save_steps=save_steps,
            placeholder_token=placeholder_token,
            placeholder_token_id=placeholder_token_id,
            save_path=output_dir,
        )

        del ti_optimizer

    # Next perform Tuning with LoRA:
    unet_lora_params, _ = inject_trainable_lora(
        unet, r=lora_rank, target_replace_module=lora_unet_target_modules
    )

    print("Before training:")
    inspect_lora(unet)

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
    ]

    text_encoder.requires_grad_(False)

    if train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            text_encoder,
            target_replace_module=lora_clip_target_modules,
            r=lora_rank,
        )
        params_to_optimize += [
            {
                "params": itertools.chain(*text_encoder_lora_params),
                "lr": text_encoder_lr,
            }
        ]
        inspect_lora(text_encoder)

    lora_optimizers = optim.AdamW(params_to_optimize, weight_decay=weight_decay_lora)

    unet.train()
    if train_text_encoder:
        text_encoder.train()

    perform_tuning(
        unet,
        vae,
        text_encoder,
        train_dataloader,
        max_train_steps_tuning,
        scheduler=noise_scheduler,
        optimizer=lora_optimizers,
        save_steps=save_steps,
        placeholder_token=placeholder_token,
        placeholder_token_id=placeholder_token_id,
        save_path=output_dir,
    )


def main():
    fire.Fire(train)

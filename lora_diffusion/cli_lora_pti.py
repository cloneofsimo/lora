# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

import argparse
import hashlib
import inspect
import itertools
import math
import os
import json
import time
import random
import re
from pathlib import Path
import numpy as np
from typing import Optional, List, Literal

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
import wandb
import fire

from lora_diffusion import (
    PivotalTuningDatasetCapation,
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    inspect_lora,
    save_lora_weight,
    save_all,
    prepare_clip_model_sets,
    evaluate_pipe,
    UNET_EXTENDED_TARGET_REPLACE,
    parse_safeloras_embeds,
    apply_learned_embed_in_clip,
)

def preview_training_batch(train_dataloader, mode, n_imgs = 40):
    outdir = f"training_batch_preview/{mode}"
    os.makedirs(outdir, exist_ok=True)
    imgs_saved = 0

    while True:
        for batch_i, batch in enumerate(train_dataloader):
            imgs = batch["pixel_values"]
            for i, img_torch in enumerate(imgs):
                img_torch = (img_torch+1) /2
                # convert to pil and save to disk:
                img = Image.fromarray((255.*img_torch).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)).convert("RGB")
                img.save(f"{outdir}/preview_{imgs_saved}.jpg")
                imgs_saved += 1

        if imgs_saved > n_imgs:
            print(f"\nSaved {imgs_saved} preview training imgs to {outdir}")
            return

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def compute_pairwise_distances(x,y):
    # compute the L2 distance of each row in x to each row in y (both are torch tensors)
    # x is a torch tensor of shape (m, d)
    # y is a torch tensor of shape (n, d)
    # returns a torch tensor of shape (m, n)

    n = y.shape[0]
    m = x.shape[0]
    d = x.shape[1]

    x = x.unsqueeze(1).expand(m, n, d)
    y = y.unsqueeze(0).expand(m, n, d)

    return torch.pow(x - y, 2).sum(2)


def print_most_similar_tokens(tokenizer, optimized_token, text_encoder, n=10):
    with torch.no_grad():
        # get all the token embeddings:
        token_embeds = text_encoder.get_input_embeddings().weight.data

        # Compute the cosine-similarity between the optimized tokens and all the other tokens
        similarity = sim_matrix(optimized_token.unsqueeze(0), token_embeds).squeeze()
        similarity = similarity.detach().cpu().numpy()

        distances = compute_pairwise_distances(optimized_token.unsqueeze(0), token_embeds).squeeze()
        distances = distances.detach().cpu().numpy()
        
        # print similarity for the most similar tokens:
        most_similar_tokens = np.argsort(similarity)[::-1]

        print(f"{tokenizer.decode(most_similar_tokens[0])} --> mean: {optimized_token.mean().item():.3f}, std: {optimized_token.std().item():.3f}, norm: {optimized_token.norm():.4f}")
        for token_id in most_similar_tokens[1:n+1]:
            print(f"sim of {similarity[token_id]:.3f} & L2 of {distances[token_id]:.3f} with \"{tokenizer.decode(token_id)}\"")


def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    placeholder_tokens: List[str],
    initializer_tokens: List[str],
    device="cuda:0",
):

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    placeholder_token_ids = []

    for token, init_tok in zip(placeholder_tokens, initializer_tokens):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids.append(placeholder_token_id)

        # Load models and create wrapper for stable diffusion

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):
            # <rand-"sigma">, e.g. <rand-0.5>
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

            # print some stats about the token embedding:
            t = token_embeds[placeholder_token_id]
            print(f"init_token {init_tok} --> mean: {t.mean().item():.3f}, std: {t.std().item():.3f}, norm: {t.norm():.4f}")


    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder=None if pretrained_vae_name_or_path else "vae",
        revision=None if pretrained_vae_name_or_path else revision,
        local_files_only = True,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
        local_files_only = True,
    )

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_ids
    )


@torch.no_grad()
def text2img_dataloader(
    train_dataset,
    train_batch_size,
    tokenizer,
    vae,
    text_encoder,
    cached_latents: bool = False,
):

    if cached_latents:
        cached_latents_dataset = []
        for idx in tqdm(range(len(train_dataset))):
            batch = train_dataset[idx]
            # rint(batch)
            latents = vae.encode(
                batch["instance_images"].unsqueeze(0).to(dtype=vae.dtype).to(vae.device)
            ).latent_dist.sample()
            latents = latents * 0.18215
            batch["instance_images"] = latents.squeeze(0)
            cached_latents_dataset.append(batch)

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
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

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    if cached_latents:

        train_dataloader = torch.utils.data.DataLoader(
            cached_latents_dataset,
            batch_size=train_batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=collate_fn,
        )

        print("PTI : Using cached latent.")

    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=collate_fn,
        )

    return train_dataloader


def inpainting_dataloader(
    train_dataset, train_batch_size, tokenizer, vae, text_encoder
):
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        mask_values = [example["instance_masks"] for example in examples]
        masked_image_values = [
            example["instance_masked_images"] for example in examples
        ]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if examples[0].get("class_prompt_ids", None) is not None:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            mask_values += [example["class_masks"] for example in examples]
            masked_image_values += [
                example["class_masked_images"] for example in examples
            ]

        pixel_values = (
            torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        )
        mask_values = (
            torch.stack(mask_values).to(memory_format=torch.contiguous_format).float()
        )
        masked_image_values = (
            torch.stack(masked_image_values)
            .to(memory_format=torch.contiguous_format)
            .float()
        )

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mask_values": mask_values,
            "masked_image_values": masked_image_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers = 4,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_dataloader


def loss_step(
    batch,
    unet,
    vae,
    text_encoder,
    scheduler,
    optimized_embeddings = None,
    train_inpainting=False,
    t_mutliplier=1.0,
    mixed_precision=False,
    mask_temperature=1.0,
    cached_latents: bool = False,
):
    weight_dtype = torch.float32
    if not cached_latents:
        latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype).to(unet.device)
        ).latent_dist.sample()
        latents = latents * 0.18215

        if train_inpainting:
            masked_image_latents = vae.encode(
                batch["masked_image_values"].to(dtype=weight_dtype).to(unet.device)
            ).latent_dist.sample()
            masked_image_latents = masked_image_latents * 0.18215
            mask = F.interpolate(
                batch["mask_values"].to(dtype=weight_dtype).to(unet.device),
                scale_factor=1 / 8,
            )
    else:
        latents = batch["pixel_values"]

        if train_inpainting:
            masked_image_latents = batch["masked_image_latents"]
            mask = batch["mask_values"]

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    if train_inpainting:
        latent_model_input = torch.cat(
            [noisy_latents, mask, masked_image_latents], dim=1
        )
    else:
        latent_model_input = noisy_latents

    if mixed_precision:
        with torch.cuda.amp.autocast():

            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(text_encoder.device)
            )[0]

            model_pred = unet(
                latent_model_input, timesteps, encoder_hidden_states
            ).sample
    else:

        encoder_hidden_states = text_encoder(
            batch["input_ids"].to(text_encoder.device)
        )[0]

        model_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

    if batch.get("mask", None) is not None:

        mask = (
            batch["mask"]
            .to(model_pred.device)
            .reshape(
                model_pred.shape[0], 1, model_pred.shape[2] * 8, model_pred.shape[3] * 8
            )
        )
        # resize to match model_pred
        mask = F.interpolate(
            mask.float(),
            size=model_pred.shape[-2:],
            mode="nearest",
        )

        mask = (mask + 0.01).pow(mask_temperature)

        mask = mask / mask.max()

        model_pred = model_pred * mask

        target = target * mask

    loss = (
        F.mse_loss(model_pred.float(), target.float(), reduction="none")
        .mean([1, 2, 3])
        .mean()
    )

    if optimized_embeddings is not None:
        embedding_norm = optimized_embeddings.norm(dim=1).mean()
        target_norm = 0.39
        embedding_norm_loss = (embedding_norm - target_norm)**2
        loss += 0.005*embedding_norm_loss

    return loss


def train_inversion(
    unet,
    vae,
    text_encoder,
    dataloader,
    num_steps: int,
    scheduler,
    index_no_updates,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path: str,
    tokenizer,
    lr_scheduler,
    test_image_path: str,
    cached_latents: bool,
    accum_iter: int = 1,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False,
    mixed_precision: bool = False,
    clip_ti_decay: bool = True,
):

    print("Performing Inversion....")
    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    # Original Emb for TI
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        preped_clip = prepare_clip_model_sets()

    index_updates = ~index_no_updates
    loss_sum = 0.0
    losses = []

    for epoch in range(math.ceil(num_steps / len(dataloader))):
        unet.eval()
        text_encoder.train()
        for batch in dataloader:

            lr_scheduler.step()

            with torch.set_grad_enabled(True):
                loss = (
                    loss_step(
                        batch,
                        unet,
                        vae,
                        text_encoder,
                        scheduler,
                        optimized_embeddings = text_encoder.get_input_embeddings().weight[index_updates, :],
                        train_inpainting=train_inpainting,
                        mixed_precision=mixed_precision,
                        cached_latents=cached_latents,
                    )
                    / accum_iter
                )

                losses.append(loss.detach().mean().item())
                loss.backward()
                loss_sum += loss.detach().item()

                if global_step % accum_iter == 0:
                    # print gradient of text encoder embedding
                    if 0:
                        print(
                            text_encoder.get_input_embeddings()
                            .weight.grad[index_updates, :]
                            .norm(dim=-1)
                            .mean()
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                    with torch.no_grad():

                        # normalize embeddings
                        if clip_ti_decay:
                            pre_norm = (
                                text_encoder.get_input_embeddings()
                                .weight[index_updates, :]
                                .norm(dim=-1, keepdim=True)
                            )

                            lambda_ = min(1.0, 100 * lr_scheduler.get_last_lr()[0])
                            text_encoder.get_input_embeddings().weight[
                                index_updates
                            ] = F.normalize(
                                text_encoder.get_input_embeddings().weight[
                                    index_updates, :
                                ],
                                dim=-1,
                            ) * (
                                pre_norm + lambda_ * (0.4 - pre_norm)
                            )
                            #print(pre_norm)

                        optimizing_embeds = text_encoder.get_input_embeddings().weight[index_updates, :]
                        current_norm = (optimizing_embeds.norm(dim=-1))

                        # reset original embeddings (we're only optimizing the new token ones)
                        text_encoder.get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]
                        
                        if global_step % 50 == 0:
                            print("------------------------------")
                            for i, t in enumerate(optimizing_embeds):
                                print_most_similar_tokens(tokenizer, t, text_encoder)

                global_step += 1
                progress_bar.update(1)

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if global_step % save_steps == 0:
                plot_loss_curve(losses, "textual_inversion")
                save_all(
                    unet=unet,
                    text_encoder=text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(
                        save_path, f"step_inv_{global_step:04d}.safetensors"
                    ),
                    save_lora=False,
                )
                if log_wandb:
                    with torch.no_grad():
                        pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                        )

                        # open all images in test_image_path
                        images = []
                        for file in os.listdir(test_image_path):
                            if (
                                file.lower().endswith(".png")
                                or file.lower().endswith(".jpg")
                                or file.lower().endswith(".jpeg")
                            ):
                                images.append(
                                    Image.open(os.path.join(test_image_path, file))
                                )

                        wandb.log({"loss": loss_sum / save_steps})
                        loss_sum = 0.0
                        wandb.log(
                            evaluate_pipe(
                                pipe,
                                target_images=images,
                                class_token=class_token,
                                learnt_token="".join(placeholder_tokens),
                                n_test=wandb_log_prompt_cnt,
                                n_step=50,
                                clip_model_sets=preped_clip,
                            )
                        )

            if global_step >= num_steps:
                return

import matplotlib.pyplot as plt
def plot_loss_curve(losses, name, moving_avg=5):
    losses = np.array(losses)
    losses = np.convolve(losses, np.ones(moving_avg)/moving_avg, mode='valid')
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Losses during {name} phase:")
    plt.savefig(f"{name}.png")
    plt.clf()

def perform_tuning(
    unet,
    vae,
    text_encoder,
    dataloader,
    num_steps,
    scheduler,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path,
    lr_scheduler_lora,
    lora_unet_target_modules,
    lora_clip_target_modules,
    mask_temperature,
    out_name: str,
    tokenizer,
    test_image_path: str,
    cached_latents: bool,
    index_no_updates = None,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False,
):
    print("Performing Tuning....")
    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    weight_dtype = torch.float16

    unet.train()
    text_encoder.train()

    # Save the current token embeddings:
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        preped_clip = prepare_clip_model_sets()

    print(f"Performing {math.ceil(num_steps / len(dataloader))} epochs of training!")
    loss_sum = 0.0
    losses = []

    for epoch in range(math.ceil(num_steps / len(dataloader))):
        if not cached_latents:
            dataloader.dataset.tune_h_flip_prob(epoch / math.ceil(num_steps / len(dataloader)))

        for batch in dataloader:
            lr_scheduler_lora.step()

            optimizer.zero_grad()

            loss = loss_step(
                batch,
                unet,
                vae,
                text_encoder,
                scheduler,
                optimized_embeddings = text_encoder.get_input_embeddings().weight[~index_no_updates, :], 
                train_inpainting=train_inpainting,
                t_mutliplier=0.8,
                mixed_precision=True,
                mask_temperature=mask_temperature,
                cached_latents=cached_latents,
            )
            loss_sum += loss.detach().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(unet.parameters(), text_encoder.parameters()), 1.0
            )
            optimizer.step()
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            losses.append(loss.detach().item())

            if index_no_updates is not None:
                with torch.no_grad():
                    # reset original embeddings (we're only optimizing the new tokens)
                    text_encoder.get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            if global_step % 100 == 0:
                optimizing_embeds = text_encoder.get_input_embeddings().weight[~index_no_updates]
                print("------------------------------")
                for i, t in enumerate(optimizing_embeds):
                    print_most_similar_tokens(tokenizer, t, text_encoder)


            global_step += 1

            if global_step % save_steps == 0:
                # plot the loss curve:
                plot_loss_curve(losses, "tuning")

                save_all(
                    unet,
                    text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(
                        save_path, f"step_{global_step:04d}.safetensors"
                    ),
                    target_replace_module_text=lora_clip_target_modules,
                    target_replace_module_unet=lora_unet_target_modules,
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

                if log_wandb:
                    with torch.no_grad():
                        pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                        )

                        # open all images in test_image_path
                        images = []
                        for file in os.listdir(test_image_path):
                            if file.endswith(".png") or file.endswith(".jpg"):
                                images.append(
                                    Image.open(os.path.join(test_image_path, file))
                                )

                        wandb.log({"loss": loss_sum / save_steps})
                        loss_sum = 0.0
                        wandb.log(
                            evaluate_pipe(
                                pipe,
                                target_images=images,
                                class_token=class_token,
                                learnt_token="".join(placeholder_tokens),
                                n_test=wandb_log_prompt_cnt,
                                n_step=50,
                                clip_model_sets=preped_clip,
                            )
                        )

            if global_step >= num_steps:
                break

    save_all(
        unet,
        text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=os.path.join(save_path, f"{out_name}.safetensors"),
        target_replace_module_text=lora_clip_target_modules,
        target_replace_module_unet=lora_unet_target_modules,
    )

def train(
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    train_text_encoder: bool = True,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    perform_inversion: bool = True,
    use_template: Literal[None, "object", "style", "person"] = None,
    train_inpainting: bool = False,
    placeholder_tokens: str = "",
    placeholder_token_at_data: Optional[str] = None,
    initializer_tokens: Optional[str] = None,
    load_pretrained_inversion_embeddings_path: Optional[str] = None,
    seed: int = 42,
    resolution: int = 512,
    color_jitter: bool = True,
    train_batch_size: int = 1,
    sample_batch_size: int = 1,
    max_train_steps_tuning: int = 1000,
    max_train_steps_ti: int = 1000,
    save_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = False,
    lora_rank_unet: int = 4,
    lora_rank_text_encoder: int = 4,
    lora_unet_target_modules={"CrossAttention", "Attention", "GEGLU"},
    lora_clip_target_modules={"CLIPAttention"},
    lora_dropout_p: float = 0.0,
    lora_scale: float = 1.0,
    use_extended_lora: bool = False,
    clip_ti_decay: bool = True,
    learning_rate_unet: float = 1e-4,
    learning_rate_text: float = 1e-5,
    learning_rate_ti: float = 5e-4,
    continue_inversion: bool = False,
    continue_inversion_lr: Optional[float] = None,
    use_face_segmentation_condition: bool = False,
    cached_latents: bool = True,
    use_mask_captioned_data: bool = False,
    mask_temperature: float = 1.0,
    scale_lr: bool = False,
    lr_scheduler: str = "linear",
    lr_warmup_steps: int = 0,
    lr_scheduler_lora: str = "linear",
    lr_warmup_steps_lora: int = 0,
    weight_decay_ti: float = 0.00,
    weight_decay_lora: float = 0.001,
    use_8bit_adam: bool = False,
    device="cuda:0",
    extra_args: Optional[dict] = None,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    wandb_project_name: str = "new_pti_project",
    wandb_entity: str = "new_pti_entity",
    proxy_token: str = "person",
    enable_xformers_memory_efficient_attention: bool = False,
    out_name: str = "final_lora",
):
    script_start_time = time.time()
    torch.manual_seed(seed)

    if use_template == "person" and not use_face_segmentation_condition:
        print("###  WARNING  ### : Using person template without face segmentation condition")
        print("When training people, it is highly recommended to use face segmentation condition!!")

    # Get a dict with all the arguments:
    args_dict = locals()

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=f"steps_{max_train_steps_ti}_lr_{learning_rate_ti}_{instance_data_dir.split('/')[-1]}",
            reinit=True,
            config={
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if len(placeholder_tokens) == 0:
        placeholder_tokens = []
        print("PTI : Placeholder Tokens not given, using null token")
    else:
        placeholder_tokens = placeholder_tokens.split("|")
        assert (
            sorted(placeholder_tokens) == placeholder_tokens
        ), f"Placeholder tokens should be sorted. Use something like {'|'.join(sorted(placeholder_tokens))}'"

    if initializer_tokens is None:
        print("PTI : Initializer Tokens not given, doing random inits")
        initializer_tokens = ["<rand-0.017>"] * len(placeholder_tokens)
    else:
        initializer_tokens = initializer_tokens.split("|")

    assert len(initializer_tokens) == len(
        placeholder_tokens
    ), "Unequal Initializer token for Placeholder tokens."

    if proxy_token is not None:
        class_token = proxy_token
    class_token = "".join(initializer_tokens)

    if placeholder_token_at_data is not None:
        tok, pat = placeholder_token_at_data.split("|")
        token_map = {tok: pat}

    else:
        token_map = {"DUMMY": "".join(placeholder_tokens)}

    print("PTI : Placeholder Tokens", placeholder_tokens)
    print("PTI : Initializer Tokens", initializer_tokens)
    print("PTI : Token Map: ", token_map)

    # get the models
    text_encoder, vae, unet, tokenizer, placeholder_token_ids = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_tokens,
        initializer_tokens,
        device=device,
    )

    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler", 
        local_files_only = True,
    )

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

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
        token_map=token_map,
        use_template=use_template,
        tokenizer=tokenizer,
        size=resolution,
        color_jitter=color_jitter,
        use_face_segmentation_condition=use_face_segmentation_condition,
        use_mask_captioned_data=use_mask_captioned_data,
        train_inpainting=train_inpainting,
    )

    if train_inpainting:
        assert not cached_latents, "Cached latents not supported for inpainting"

        train_dataloader = inpainting_dataloader(
            train_dataset, train_batch_size, tokenizer, vae, text_encoder
        )
    else:
        train_dataloader = text2img_dataloader(
            train_dataset,
            train_batch_size,
            tokenizer,
            vae,
            text_encoder,
            cached_latents=cached_latents,
        )

    index_no_updates = torch.arange(len(tokenizer)) != -1

    for tok_id in placeholder_token_ids:
        index_no_updates[tok_id] = False

    unet.requires_grad_(False)
    vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    if cached_latents:
        vae = None

    # STEP 1 : Perform Inversion
    if perform_inversion and not cached_latents and (load_pretrained_inversion_embeddings_path is None):
        preview_training_batch(train_dataloader, "inversion")

        print("PTI : Performing Inversion")
        ti_optimizer = optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),
            lr=ti_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay_ti,
        )

        token_ids_positions_to_update = np.where(index_no_updates.cpu().numpy() == 0)
        print("Training embedding of size", text_encoder.get_input_embeddings().weight[token_ids_positions_to_update].shape)

        lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=ti_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps_ti,
        )

        train_inversion(
            unet,
            vae,
            text_encoder,
            train_dataloader,
            max_train_steps_ti,
            cached_latents=cached_latents,
            accum_iter=gradient_accumulation_steps,
            scheduler=noise_scheduler,
            index_no_updates=index_no_updates,
            optimizer=ti_optimizer,
            lr_scheduler=lr_scheduler,
            save_steps=save_steps,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_ids=placeholder_token_ids,
            save_path=output_dir,
            test_image_path=instance_data_dir,
            log_wandb=log_wandb,
            wandb_log_prompt_cnt=wandb_log_prompt_cnt,
            class_token=class_token,
            train_inpainting=train_inpainting,
            mixed_precision=False,
            tokenizer=tokenizer,
            clip_ti_decay=clip_ti_decay,
        )

        del ti_optimizer
        print("###############  Inversion Done  ###############")

    elif load_pretrained_inversion_embeddings_path is not None:

        print("PTI : Loading pretrained inversion embeddings..")
        from safetensors.torch import safe_open
        # Load the pretrained embeddings from the lora file:
        safeloras = safe_open(load_pretrained_inversion_embeddings_path, framework="pt", device="cpu")
        #monkeypatch_or_replace_safeloras(pipe, safeloras)
        tok_dict = parse_safeloras_embeds(safeloras)
        apply_learned_embed_in_clip(
                tok_dict,
                text_encoder,
                tokenizer,
                idempotent=True,
            )

    # Next perform Tuning with LoRA:
    if not use_extended_lora:
        unet_lora_params, _ = inject_trainable_lora(
            unet,
            r=lora_rank_unet,
            target_replace_module=lora_unet_target_modules,
            dropout_p=lora_dropout_p,
            scale=lora_scale,
        )
        print("PTI : not use_extended_lora...")
        print("PTI : Will replace modules: ", lora_unet_target_modules)
    else:
        print("PTI : USING EXTENDED UNET!!!")
        lora_unet_target_modules = (
            lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
        )
        print("PTI : Will replace modules: ", lora_unet_target_modules)
        unet_lora_params, _ = inject_trainable_lora_extended(
            unet, r=lora_rank_unet, target_replace_module=lora_unet_target_modules
        )

    #n_optimizable_unet_params = sum([el.numel() for el in itertools.chain(*unet_lora_params)])
    #print("PTI : Number of optimizable UNET parameters: ", n_optimizable_unet_params)

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
    ]

    text_encoder.requires_grad_(False)

    if continue_inversion:
        params_to_optimize += [
            {
                "params": text_encoder.get_input_embeddings().parameters(),
                "lr": continue_inversion_lr
                if continue_inversion_lr is not None
                else ti_lr,
            }
        ]
        text_encoder.requires_grad_(True)
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        for param in params_to_freeze:
            param.requires_grad = False
    else:
        text_encoder.requires_grad_(False)

    if train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            text_encoder,
            target_replace_module=lora_clip_target_modules,
            r=lora_rank_text_encoder,
        )
        params_to_optimize += [
            {"params": itertools.chain(*text_encoder_lora_params),
                "lr": text_encoder_lr}
        ]

        #n_optimizable_text_Encoder_params = sum( [el.numel() for el in itertools.chain(*text_encoder_lora_params)])
        #print("PTI : Number of optimizable text-encoder parameters: ", n_optimizable_text_Encoder_params)

    lora_optimizers = optim.AdamW(params_to_optimize, weight_decay=weight_decay_lora)

    unet.train()
    if train_text_encoder:
        print("Training text encoder!")
        text_encoder.train()

    lr_scheduler_lora = get_scheduler(
        lr_scheduler_lora,
        optimizer=lora_optimizers,
        num_warmup_steps=lr_warmup_steps_lora,
        num_training_steps=max_train_steps_tuning,
    )
    if not cached_latents: 
        preview_training_batch(train_dataloader, "tuning")

    #print("PTI : n_optimizable_unet_params: ", n_optimizable_unet_params)
    print(f"PTI : has {len(unet_lora_params)} lora")
    print("PTI : Before training:")

    moved = (
        torch.tensor(list(itertools.chain(*inspect_lora(unet).values())))
        .mean().item())
    print(f"LORA Unet Moved {moved:.6f}")


    moved = (
        torch.tensor(
            list(itertools.chain(*inspect_lora(text_encoder).values()))
        ).mean().item())
    print(f"LORA CLIP Moved {moved:.6f}")

    perform_tuning(
        unet,
        vae,
        text_encoder,
        train_dataloader,
        max_train_steps_tuning,
        index_no_updates = index_no_updates,
        cached_latents=cached_latents,
        scheduler=noise_scheduler,
        optimizer=lora_optimizers,
        save_steps=save_steps,
        placeholder_tokens=placeholder_tokens,
        placeholder_token_ids=placeholder_token_ids,
        save_path=output_dir,
        lr_scheduler_lora=lr_scheduler_lora,
        lora_unet_target_modules=lora_unet_target_modules,
        lora_clip_target_modules=lora_clip_target_modules,
        mask_temperature=mask_temperature,
        tokenizer=tokenizer,
        out_name=out_name,
        test_image_path=instance_data_dir,
        log_wandb=log_wandb,
        wandb_log_prompt_cnt=wandb_log_prompt_cnt,
        class_token=class_token,
        train_inpainting=train_inpainting,
    )

    print("###############  Tuning Done  ###############")
    training_time = time.time() - script_start_time
    print(f"Training time: {training_time/60:.1f} minutes")
    args_dict["training_time_s"] = int(training_time)
    args_dict["n_epochs"] = math.ceil(max_train_steps_tuning / len(train_dataloader.dataset))
    args_dict["n_training_imgs"] = len(train_dataloader.dataset)

    # Save the args_dict to the output directory as a json file:
    with open(os.path.join(output_dir, "lora_training_args.json"), "w") as f:
        json.dump(args_dict, f, default=lambda o: '<not serializable>', indent=2)

def main():
    fire.Fire(train)

if __name__ == "__main__":
    main()
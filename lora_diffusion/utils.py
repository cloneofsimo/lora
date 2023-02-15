from typing import List, Union

import torch
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import StableDiffusionPipeline
from .lora import patch_pipe, tune_lora_scale, _text_lora_path, _ti_lora_path
import os
import glob
import math

EXAMPLE_PROMPTS = [
    "<obj> swimming in a pool",
    "<obj> at a beach with a view of seashore",
    "<obj> in times square",
    "<obj> wearing sunglasses",
    "<obj> in a construction outfit",
    "<obj> playing with a ball",
    "<obj> wearing headphones",
    "<obj> oil painting ghibli inspired",
    "<obj> working on the laptop",
    "<obj> with mountains and sunset in background",
    "Painting of <obj> at a beach by artist claude monet",
    "<obj> digital painting 3d render geometric style",
    "A screaming <obj>",
    "A depressed <obj>",
    "A sleeping <obj>",
    "A sad <obj>",
    "A joyous <obj>",
    "A frowning <obj>",
    "A sculpture of <obj>",
    "<obj> near a pool",
    "<obj> at a beach with a view of seashore",
    "<obj> in a garden",
    "<obj> in grand canyon",
    "<obj> floating in ocean",
    "<obj> and an armchair",
    "A maple tree on the side of <obj>",
    "<obj> and an orange sofa",
    "<obj> with chocolate cake on it",
    "<obj> with a vase of rose flowers on it",
    "A digital illustration of <obj>",
    "Georgia O'Keeffe style <obj> painting",
    "A watercolor painting of <obj> on a beach",
]


def image_grid(_imgs, rows=None, cols=None):

    if rows is None and cols is None:
        rows = cols = math.ceil(len(_imgs) ** 0.5)

    if rows is None:
        rows = math.ceil(len(_imgs) / cols)
    if cols is None:
        cols = math.ceil(len(_imgs) / rows)

    w, h = _imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(_imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def text_img_alignment(img_embeds, text_embeds, target_img_embeds):
    # evaluation inspired from textual inversion paper
    # https://arxiv.org/abs/2208.01618

    # text alignment
    assert img_embeds.shape[0] == text_embeds.shape[0]
    text_img_sim = (img_embeds * text_embeds).sum(dim=-1) / (
        img_embeds.norm(dim=-1) * text_embeds.norm(dim=-1)
    )

    # image alignment
    img_embed_normalized = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

    avg_target_img_embed = (
        (target_img_embeds / target_img_embeds.norm(dim=-1, keepdim=True))
        .mean(dim=0)
        .unsqueeze(0)
        .repeat(img_embeds.shape[0], 1)
    )

    img_img_sim = (img_embed_normalized * avg_target_img_embed).sum(dim=-1)

    return {
        "text_alignment_avg": text_img_sim.mean().item(),
        "image_alignment_avg": img_img_sim.mean().item(),
        "text_alignment_all": text_img_sim.tolist(),
        "image_alignment_all": img_img_sim.tolist(),
    }


def prepare_clip_model_sets(eval_clip_id: str = "openai/clip-vit-large-patch14"):
    text_model = CLIPTextModelWithProjection.from_pretrained(eval_clip_id)
    tokenizer = CLIPTokenizer.from_pretrained(eval_clip_id)
    vis_model = CLIPVisionModelWithProjection.from_pretrained(eval_clip_id)
    processor = CLIPProcessor.from_pretrained(eval_clip_id)

    return text_model, tokenizer, vis_model, processor


def evaluate_pipe(
    pipe,
    target_images: List[Image.Image],
    class_token: str = "",
    learnt_token: str = "",
    guidance_scale: float = 5.0,
    seed=0,
    clip_model_sets=None,
    eval_clip_id: str = "openai/clip-vit-large-patch14",
    n_test: int = 10,
    n_step: int = 50,
):

    if clip_model_sets is not None:
        text_model, tokenizer, vis_model, processor = clip_model_sets
    else:
        text_model, tokenizer, vis_model, processor = prepare_clip_model_sets(
            eval_clip_id
        )

    images = []
    img_embeds = []
    text_embeds = []
    for prompt in EXAMPLE_PROMPTS[:n_test]:
        prompt = prompt.replace("<obj>", learnt_token)
        torch.manual_seed(seed)
        with torch.autocast("cuda"):
            img = pipe(
                prompt, num_inference_steps=n_step, guidance_scale=guidance_scale
            ).images[0]
        images.append(img)

        # image
        inputs = processor(images=img, return_tensors="pt")
        img_embed = vis_model(**inputs).image_embeds
        img_embeds.append(img_embed)

        prompt = prompt.replace(learnt_token, class_token)
        # prompts
        inputs = tokenizer([prompt], padding=True, return_tensors="pt")
        outputs = text_model(**inputs)
        text_embed = outputs.text_embeds
        text_embeds.append(text_embed)

    # target images
    inputs = processor(images=target_images, return_tensors="pt")
    target_img_embeds = vis_model(**inputs).image_embeds

    img_embeds = torch.cat(img_embeds, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)

    return text_img_alignment(img_embeds, text_embeds, target_img_embeds)


def visualize_progress(
    path_alls: Union[str, List[str]],
    prompt: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device="cuda:0",
    patch_unet=True,
    patch_text=True,
    patch_ti=True,
    unet_scale=1.0,
    text_sclae=1.0,
    num_inference_steps=50,
    guidance_scale=5.0,
    offset: int = 0,
    limit: int = 10,
    seed: int = 0,
):

    imgs = []
    if isinstance(path_alls, str):
        alls = list(set(glob.glob(path_alls)))

        alls.sort(key=os.path.getmtime)
    else:
        alls = path_alls

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)

    print(f"Found {len(alls)} checkpoints")
    for path in alls[offset:limit]:
        print(path)

        patch_pipe(
            pipe, path, patch_unet=patch_unet, patch_text=patch_text, patch_ti=patch_ti
        )

        tune_lora_scale(pipe.unet, unet_scale)
        tune_lora_scale(pipe.text_encoder, text_sclae)

        torch.manual_seed(seed)
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        imgs.append(image)

    return imgs

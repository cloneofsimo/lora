from typing import List

import torch
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

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


def image_grid(_imgs, rows, cols):

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
    avg_img_embed = (img_embeds / img_embeds.norm(dim=-1, keepdim=True)).mean(dim=0)
    avg_target_img_embed = (
        target_img_embeds / target_img_embeds.norm(dim=-1, keepdim=True)
    ).mean(dim=0)

    img_img_sim = (avg_img_embed * avg_target_img_embed).sum() / (
        avg_img_embed.norm() * avg_target_img_embed.norm()
    )

    return {
        "text_alignment_avg": text_img_sim.mean().item(),
        "image_alignment_avg": img_img_sim.item(),
        "text_alignment_all": text_img_sim.tolist(),
    }


def evaluate_pipe(
    pipe,
    target_images: List[Image.Image],
    class_token: str = "",
    guidance_scale: float = 5.0,
    seed=0,
    clip_model_sets=None,
    eval_clip_id: str = "openai/clip-vit-large-patch14",
    n_test: int = 10,
):

    if clip_model_sets is not None:
        text_model, tokenizer, vis_model, processor = clip_model_sets
    else:

        text_model = CLIPTextModelWithProjection.from_pretrained(eval_clip_id)
        tokenizer = CLIPTokenizer.from_pretrained(eval_clip_id)
        vis_model = CLIPVisionModelWithProjection.from_pretrained(eval_clip_id)
        processor = CLIPProcessor.from_pretrained(eval_clip_id)

    images = []
    img_embeds = []
    text_embeds = []
    for prompt in EXAMPLE_PROMPTS[:n_test]:
        prompt = prompt.replace("<obj>", class_token)
        torch.manual_seed(seed)
        img = pipe(
            prompt, num_inference_steps=50, guidance_scale=guidance_scale
        ).images[0]
        images.append(img)

        # image
        inputs = processor(images=img, return_tensors="pt")
        img_embed = vis_model(**inputs).image_embeds
        img_embeds.append(img_embed)

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
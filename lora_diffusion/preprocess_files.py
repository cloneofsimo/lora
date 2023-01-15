# Have SwinIR upsample
# Have BLIP auto caption
# Have CLIPSeg auto mask concept

from typing import List, Literal, Union
import os
from PIL import Image
import torch
import numpy as np
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


@torch.no_grad()
def swin_ir_sr(
    images: List[Image.Image],
    model_id: Literal[
        "caidas/swin2SR-classical-sr-x2-64", "caidas/swin2SR-classical-sr-x4-48"
    ] = "caidas/swin2SR-classical-sr-x2-64",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    **kwargs,
) -> None:
    # So this is currently in main branch, so this can be used in the future I guess?
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    model = Swin2SRForImageSuperResolution.from_pretrained(
        model_id,
    ).to(device)
    processor = Swin2SRImageProcessor()

    out_images = []

    for image in images:

        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        output = (
            outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        )
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)
        output = Image.fromarray(output)

        out_images.append(output)

    return out_images


@torch.no_grad()
def clipseg_mask_generator(
    images: List[Image.Image],
    target_prompts: Union[List[str], str],
    model_id: Literal[
        "CIDAS/clipseg-rd64-refined", "CIDAS/clipseg-rd16"
    ] = "CIDAS/clipseg-rd64-refined",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    bias: float = 0.05,
    **kwargs,
):

    if isinstance(target_prompts, str):
        print(
            f'Warning: only one target prompt "{target_prompts}" was given, so it will be used for all images'
        )

        target_prompts = [target_prompts] * len(images)

    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(device)

    masks = []

    for image, prompt in zip(images, target_prompts):

        original_size = image.size

        inputs = processor(
            text=[prompt, ""],
            images=[image] * 2,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=0)[0]
        probs = (probs + bias).clamp_(0, 1)
        probs = 255 * probs / probs.max()

        # make mask greyscale
        mask = Image.fromarray(probs.cpu().numpy()).convert("L")

        # resize mask to original size
        mask = mask.resize(original_size)

        masks.append(mask)

    return masks


@torch.no_grad()
def blip_captioning_dataset(
    images: List[Image.Image],
    model_id: Literal[
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-image-captioning-base",
    ] = "Salesforce/blip-image-captioning-base",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    **kwargs,
):

    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    captions = []

    for image in images:
        inputs = processor(image, return_tensors="pt").to("cuda")

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        captions.append(caption)

    return captions

# Have SwinIR upsample
# Have BLIP auto caption
# Have CLIPSeg auto mask concept

from typing import List, Literal, Union, Optional, Tuple
import os
from PIL import Image, ImageFilter
import torch
import numpy as np
import fire
from tqdm import tqdm
import glob
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


@torch.no_grad()
def swin_ir_sr(
    images: List[Image.Image],
    model_id: Literal[
        "caidas/swin2SR-classical-sr-x2-64", "caidas/swin2SR-classical-sr-x4-48"
    ] = "caidas/swin2SR-classical-sr-x2-64",
    target_size: Optional[Tuple[int, int]] = None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    **kwargs,
) -> List[Image.Image]:
    """
    Upscales images using SwinIR. Returns a list of PIL images.
    """
    # So this is currently in main branch, so this can be used in the future I guess?
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    model = Swin2SRForImageSuperResolution.from_pretrained(
        model_id,
    ).to(device)
    processor = Swin2SRImageProcessor()

    out_images = []

    for image in tqdm(images):

        ori_w, ori_h = image.size
        if target_size is not None:
            if ori_w >= target_size[0] and ori_h >= target_size[1]:
                out_images.append(image)
                continue

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
    bias: float = 0.01,
    temp: float = 1.0,
    **kwargs,
) -> List[Image.Image]:
    """
    Returns a greyscale mask for each image, where the mask is the probability of the target prompt being present in the image
    """

    if isinstance(target_prompts, str):
        print(
            f'Warning: only one target prompt "{target_prompts}" was given, so it will be used for all images'
        )

        target_prompts = [target_prompts] * len(images)

    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(device)

    masks = []

    for image, prompt in tqdm(zip(images, target_prompts)):

        original_size = image.size

        inputs = processor(
            text=[prompt, ""],
            images=[image] * 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits / temp, dim=0)[0]
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
    text: Optional[str] = None,
    model_id: Literal[
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-image-captioning-base",
    ] = "Salesforce/blip-image-captioning-large",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    **kwargs,
) -> List[str]:
    """
    Returns a list of captions for the given images
    """

    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    captions = []

    for image in tqdm(images):
        inputs = processor(image, text=text, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs, max_length=150, do_sample=True, top_k=50, temperature=0.7
        )
        caption = processor.decode(out[0], skip_special_tokens=True)

        captions.append(caption)

    return captions

def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 80.0, bias: float = 0.05
) -> List[Image.Image]:
    """
    Returns a list of images with mask on the face parts.
    """
    import mediapipe as mp

    mp_face_detection = mp.solutions.face_detection

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    masks = []
    for image in tqdm(images):

        image = np.array(image)

        results = face_detection.process(image)
        black_image = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        if results.detections:

            for detection in results.detections:

                x_min = int(
                    detection.location_data.relative_bounding_box.xmin * image.shape[1]
                )
                y_min = int(
                    detection.location_data.relative_bounding_box.ymin * image.shape[0]
                )
                width = int(
                    detection.location_data.relative_bounding_box.width * image.shape[1]
                )
                height = int(
                    detection.location_data.relative_bounding_box.height
                    * image.shape[0]
                )

                # draw the colored rectangle
                black_image[y_min : y_min + height, x_min : x_min + width] = 255

        black_image = Image.fromarray(black_image)
        masks.append(black_image)

    return masks


def _crop_to_square(
    image: Image.Image, com: List[Tuple[int, int]], resize_to: Optional[int] = None
):
    cx, cy = com
    width, height = image.size
    if width > height:
        left_possible = max(cx - height / 2, 0)
        left = min(left_possible, width - height)
        right = left + height
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top_possible = max(cy - width / 2, 0)
        top = min(top_possible, height - width)
        bottom = top + width

    image = image.crop((left, top, right, bottom))

    if resize_to:
        image = image.resize((resize_to, resize_to), Image.Resampling.LANCZOS)

    return image


def _center_of_mass(mask: Image.Image):
    """
    Returns the center of mass of the mask
    """
    x, y = np.meshgrid(np.arange(mask.size[0]), np.arange(mask.size[1]))

    x_ = x * np.array(mask)
    y_ = y * np.array(mask)

    x = np.sum(x_) / np.sum(mask)
    y = np.sum(y_) / np.sum(mask)

    return x, y


def load_and_save_masks_and_captions(
    files: Union[str, List[str]],
    output_dir: str,
    caption_text: Optional[str] = None,
    target_prompts: Optional[Union[List[str], str]] = None,
    target_size: int = 512,
    crop_based_on_salience: bool = True,
    use_face_detection_instead: bool = False,
    temp: float = 1.0,
    n_length: int = -1,
):
    """
    Loads images from the given files, generates masks for them, and saves the masks and captions and upscale images
    to output dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # load images
    if isinstance(files, str):
        # check if it is a directory
        if os.path.isdir(files):
            # get all the .png .jpg in the directory
            files = glob.glob(os.path.join(files, "*.png")) + glob.glob(os.path.join(files, "*.jpg")) + glob.glob(os.path.join(files, "*.jpeg"))

        if len(files) == 0:
            raise Exception(
                f"No files found in {files}. Either {files} is not a directory or it does not contain any .png or .jpg files."
            )
        if n_length == -1:
            n_length = len(files)
        files = sorted(files)[:n_length]

    images = [Image.open(file) for file in files]

    # captions
    print(f"Generating {len(images)} captions...")
    captions = blip_captioning_dataset(images, text=caption_text)

    if target_prompts is None:
        target_prompts = captions

    print(f"Generating {len(images)} masks...")
    if not use_face_detection_instead:
        seg_masks = clipseg_mask_generator(
            images=images, target_prompts=target_prompts, temp=temp
        )
    else:
        seg_masks = face_mask_google_mediapipe(images=images)

    # find the center of mass of the mask
    if crop_based_on_salience:
        coms = [_center_of_mass(mask) for mask in seg_masks]
    else:
        coms = [(image.size[0] / 2, image.size[1] / 2) for image in images]
    # based on the center of mass, crop the image to a square
    images = [
        _crop_to_square(image, com, resize_to=None) for image, com in zip(images, coms)
    ]

    print(f"Upscaling {len(images)} images...")
    # upscale images anyways
    images = swin_ir_sr(images, target_size=(target_size, target_size))
    images = [
        image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        for image in images
    ]

    seg_masks = [
        _crop_to_square(mask, com, resize_to=target_size)
        for mask, com in zip(seg_masks, coms)
    ]
    with open(os.path.join(output_dir, "caption.txt"), "w") as f:
        # save images and masks
        for idx, (image, mask, caption) in enumerate(zip(images, seg_masks, captions)):
            image.save(os.path.join(output_dir, f"{idx}.src.jpg"), quality=99)
            mask.save(os.path.join(output_dir, f"{idx}.mask.png"))

            f.write(caption + "\n")


def main():
    fire.Fire(load_and_save_masks_and_captions)


if __name__ == "__main__":
    main()
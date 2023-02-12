import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image
from torch import zeros_like
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from .preprocess_files import face_mask_google_mediapipe

OBJECT_TEMPLATE = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

STYLE_TEMPLATE = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

NULL_TEMPLATE = ["{}"]

TEMPLATE_MAP = {
    "object": OBJECT_TEMPLATE,
    "style": STYLE_TEMPLATE,
    "null": NULL_TEMPLATE,
}


def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret


def _shuffle(lis):

    return random.sample(lis, len(lis))


def _get_cutout_holes(
    height,
    width,
    min_holes=8,
    max_holes=32,
    min_height=16,
    max_height=128,
    min_width=16,
    max_width=128,
):
    holes = []
    for _n in range(random.randint(min_holes, max_holes)):
        hole_height = random.randint(min_height, max_height)
        hole_width = random.randint(min_width, max_width)
        y1 = random.randint(0, height - hole_height)
        x1 = random.randint(0, width - hole_width)
        y2 = y1 + hole_height
        x2 = x1 + hole_width
        holes.append((x1, y1, x2, y2))
    return holes


def _generate_random_mask(image):
    mask = zeros_like(image[:1])
    holes = _get_cutout_holes(mask.shape[1], mask.shape[2])
    for (x1, y1, x2, y2) in holes:
        mask[:, y1:y2, x1:x2] = 1.0
    if random.uniform(0, 1) < 0.25:
        mask.fill_(1.0)
    masked_image = image * (mask < 0.5)
    return mask, masked_image


class PivotalTuningDatasetCapation(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        token_map: Optional[dict] = None,
        use_template: Optional[str] = None,
        size=512,
        h_flip=True,
        color_jitter=False,
        resize=True,
        use_mask_captioned_data=False,
        use_face_segmentation_condition=False,
        train_inpainting=False,
        blur_amount: int = 70,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.resize = resize
        self.train_inpainting = train_inpainting

        instance_data_root = Path(instance_data_root)
        if not instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = []
        self.mask_path = []

        assert not (
            use_mask_captioned_data and use_template
        ), "Can't use both mask caption data and template."

        # Prepare the instance images
        if use_mask_captioned_data:
            src_imgs = glob.glob(str(instance_data_root) + "/*src.jpg")
            for f in src_imgs:
                idx = int(str(Path(f).stem).split(".")[0])
                mask_path = f"{instance_data_root}/{idx}.mask.png"

                if Path(mask_path).exists():
                    self.instance_images_path.append(f)
                    self.mask_path.append(mask_path)
                else:
                    print(f"Mask not found for {f}")

            self.captions = open(f"{instance_data_root}/caption.txt").readlines()

        else:
            possibily_src_images = (
                glob.glob(str(instance_data_root) + "/*.jpg")
                + glob.glob(str(instance_data_root) + "/*.png")
                + glob.glob(str(instance_data_root) + "/*.jpeg")
            )
            possibily_src_images = (
                set(possibily_src_images)
                - set(glob.glob(str(instance_data_root) + "/*mask.png"))
                - set([str(instance_data_root) + "/caption.txt"])
            )

            self.instance_images_path = list(set(possibily_src_images))
            self.captions = [
                x.split("/")[-1].split(".")[0] for x in self.instance_images_path
            ]

        assert (
            len(self.instance_images_path) > 0
        ), "No images found in the instance data root."

        self.instance_images_path = sorted(self.instance_images_path)

        self.use_mask = use_face_segmentation_condition or use_mask_captioned_data
        self.use_mask_captioned_data = use_mask_captioned_data

        if use_face_segmentation_condition:

            for idx in range(len(self.instance_images_path)):
                targ = f"{instance_data_root}/{idx}.mask.png"
                # see if the mask exists
                if not Path(targ).exists():
                    print(f"Mask not found for {targ}")

                    print(
                        "Warning : this will pre-process all the images in the instance data root."
                    )

                    if len(self.mask_path) > 0:
                        print(
                            "Warning : masks already exists, but will be overwritten."
                        )

                    masks = face_mask_google_mediapipe(
                        [
                            Image.open(f).convert("RGB")
                            for f in self.instance_images_path
                        ]
                    )
                    for idx, mask in enumerate(masks):
                        mask.save(f"{instance_data_root}/{idx}.mask.png")

                    break

            for idx in range(len(self.instance_images_path)):
                self.mask_path.append(f"{instance_data_root}/{idx}.mask.png")

        self.num_instance_images = len(self.instance_images_path)
        self.token_map = token_map

        self.use_template = use_template
        if use_template is not None:
            self.templates = TEMPLATE_MAP[use_template]

        self._length = self.num_instance_images

        self.h_flip = h_flip
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(0.1, 0.1)
                if color_jitter
                else transforms.Lambda(lambda x: x),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.blur_amount = blur_amount

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.train_inpainting:
            (
                example["instance_masks"],
                example["instance_masked_images"],
            ) = _generate_random_mask(example["instance_images"])

        if self.use_template:
            assert self.token_map is not None
            input_tok = list(self.token_map.values())[0]

            text = random.choice(self.templates).format(input_tok)
        else:
            text = self.captions[index % self.num_instance_images].strip()

            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text = text.replace(token, value)

        print(text)

        if self.use_mask:
            example["mask"] = (
                self.image_transforms(
                    Image.open(self.mask_path[index % self.num_instance_images])
                )
                * 0.5
                + 1.0
            )

        if self.h_flip and random.random() > 0.5:
            hflip = transforms.RandomHorizontalFlip(p=1)

            example["instance_images"] = hflip(example["instance_images"])
            if self.use_mask:
                example["mask"] = hflip(example["mask"])

        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example

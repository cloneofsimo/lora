# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

import argparse
import hashlib
import itertools
import math
import os
import inspect
from pathlib import Path
from typing import Optional
import random

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras_with_embeds,
)
from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path

import random
import re


imagenet_templates_small = [
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

imagenet_style_templates_small = [
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


def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret


def _shuffle(lis):

    return random.sample(lis, len(lis))


class DreamBoothTiDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        learnable_property,
        placeholder_token,
        stochastic_attribute,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        resize=False,
        h_flip=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)

        self.placeholder_token = placeholder_token
        self.stochastic_attribute = (
            stochastic_attribute.split(",") if stochastic_attribute else []
        )

        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )

        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        if resize:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(size)
                    if center_crop
                    else transforms.Lambda(lambda x: x),
                    transforms.ColorJitter(0.2, 0.1)
                    if color_jitter
                    else transforms.Lambda(lambda x: x),
                    transforms.RandomHorizontalFlip()
                    if h_flip
                    else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.CenterCrop(size)
                    if center_crop
                    else transforms.Lambda(lambda x: x),
                    transforms.ColorJitter(0.2, 0.1)
                    if color_jitter
                    else transforms.Lambda(lambda x: x),
                    transforms.RandomHorizontalFlip()
                    if h_flip
                    else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

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

        text = random.choice(self.templates).format(
            ", ".join(
                [self.placeholder_token]
                + _shuffle(_randomset(self.stochastic_attribute))
            )
        )
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_id]
    )
    print("Current Learned Embeddings: ", learned_embeds[:4])
    print("saved to ", save_path)
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--stochastic_attribute",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="both",
        help="The output format of the model predicitions and checkpoints.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_ti",
        type=float,
        default=5e-4,
        help="Initial learning rate for embedding of textual inversion (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="object",
        required=True,
        help="Conecpt to learn : style or object?",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as initializer word.",
    )
    parser.add_argument(
        "--unfreeze_lora_step",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--just_ti",
        action="store_true",
        help="Debug to see just ti",
    )
    parser.add_argument(
        "--resize",
        type=bool,
        default=True,
        required=False,
        help="Should images be resized to --resolution before training?",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            logger.warning(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    if not safetensors_available:
        if args.output_format == "both":
            print(
                "Safetensors is not available - changing output format to just output PyTorch files"
            )
            args.output_format = "pt"
        elif args.output_format == "safe":
            raise ValueError(
                "Safetensors is not available - either install it, or change output_format."
            )

    return args


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = (
                torch.float16 if accelerator.device.type == "cuda" else torch.float32
            )
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )

    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    text_encoder.resize_token_embeddings(len(tokenizer))
    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    unet.requires_grad_(False)
    unet_lora_params, _ = inject_trainable_lora(unet, r=args.lora_rank)

    for _up, _down in extract_lora_ups_down(unet):
        print("Before training: Unet First Layer lora up", _up.weight.data)
        print("Before training: Unet First Layer lora down", _down.weight.data)
        break

    vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    text_encoder_lora_params, _ = inject_trainable_lora(
        text_encoder,
        target_replace_module=["CLIPAttention"],
        r=args.lora_rank,
    )
    for _up, _down in extract_lora_ups_down(
        text_encoder, target_replace_module=["CLIPAttention"]
    ):
        print("Before training: text encoder First Layer lora up", _up.weight.data)
        print("Before training: text encoder First Layer lora down", _down.weight.data)
        break

    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
        {
            "params": itertools.chain(*text_encoder_lora_params),
            "lr": args.learning_rate_text,
        },
        {
            "params": text_encoder.get_input_embeddings().parameters(),
            "lr": args.learning_rate_ti,
        },
    ]

    if args.just_ti:
        params_to_optimize = [
            {
                "params": text_encoder.get_input_embeddings().parameters(),
                "lr": args.learning_rate_ti,
            }
        ]

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = DreamBoothTiDataset(
        instance_data_root=args.instance_data_dir,
        placeholder_token=args.placeholder_token,
        stochastic_attribute=args.stochastic_attribute,
        learnable_property=args.learnable_property,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        color_jitter=args.color_jitter,
        resize=args.resize,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
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
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    (
        unet,
        text_encoder,
        train_dataloader,
    ) = accelerator.prepare(unet, text_encoder, train_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            # freeze unet and text encoder during ti training
            if global_step < args.unfreeze_lora_step:
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = 0.0
            else:  # begin learning with unet and text encoder
                optimizer.param_groups[0]["lr"] = args.learning_rate
                optimizer.param_groups[1]["lr"] = args.learning_rate_text
                optimizer.param_groups[2]["lr"] = 0.0  # stop learning ti

            # Convert images to latent space
            latents = vae.encode(
                batch["pixel_values"].to(dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            if args.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = (
                    F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                # Compute prior loss
                prior_loss = F.mse_loss(
                    model_pred_prior.float(), target_prior.float(), reduction="mean"
                )

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder.parameters())
                    if args.train_text_encoder
                    else unet.parameters()
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            optimizer.zero_grad()

            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]

            global_step += 1

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.save_steps and global_step - last_save >= args.save_steps:
                    if accelerator.is_main_process:
                        # newer versions of accelerate allow the 'keep_fp32_wrapper' arg. without passing
                        # it, the models will be unwrapped, and when they are then used for further training,
                        # we will crash. pass this, but only to newer versions of accelerate. fixes
                        # https://github.com/huggingface/diffusers/issues/1566
                        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(
                                accelerator.unwrap_model
                            ).parameters.keys()
                        )
                        extra_args = (
                            {"keep_fp32_wrapper": True}
                            if accepts_keep_fp32_wrapper
                            else {}
                        )
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            text_encoder=accelerator.unwrap_model(
                                text_encoder, **extra_args
                            ),
                            revision=args.revision,
                        )

                        filename_unet = (
                            f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.pt"
                        )
                        filename_text_encoder = f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                        print(f"save weights {filename_unet}, {filename_text_encoder}")
                        save_lora_weight(pipeline.unet, filename_unet)

                        save_lora_weight(
                            pipeline.text_encoder,
                            filename_text_encoder,
                            target_replace_module=["CLIPAttention"],
                        )

                        for _up, _down in extract_lora_ups_down(pipeline.unet):
                            print(
                                "First Unet Layer's Up Weight is now : ",
                                _up.weight.data,
                            )
                            print(
                                "First Unet Layer's Down Weight is now : ",
                                _down.weight.data,
                            )
                            break

                        for _up, _down in extract_lora_ups_down(
                            pipeline.text_encoder,
                            target_replace_module=["CLIPAttention"],
                        ):
                            print(
                                "First Text Encoder Layer's Up Weight is now : ",
                                _up.weight.data,
                            )
                            print(
                                "First Text Encoder Layer's Down Weight is now : ",
                                _down.weight.data,
                            )
                            break

                        filename_ti = f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.ti.pt"

                        save_progress(
                            pipeline.text_encoder,
                            placeholder_token_id,
                            accelerator,
                            args,
                            filename_ti,
                        )

                        last_save = global_step

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
    accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )

        print("\n\nLora TRAINING DONE!\n\n")

        if args.output_format == "pt" or args.output_format == "both":
            save_lora_weight(pipeline.unet, args.output_dir + "/lora_weight.pt")

            save_lora_weight(
                text_encoder,
                args.output_dir + "/lora_weight.text_encoder.pt",
                target_replace_module=["CLIPAttention"],
            )

        if args.output_format == "safe" or args.output_format == "both":
            loras = {}
            loras["unet"] = (pipeline.unet, {"CrossAttention", "Attention", "GEGLU"})
            loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

            learned_embeds = (
                accelerator.unwrap_model(text_encoder)
                .get_input_embeddings()
                .weight[placeholder_token_id]
            )

            embeds = {args.placeholder_token: learned_embeds.detach().cpu()}

            save_safeloras_with_embeds(
                loras, embeds, args.output_dir + "/lora_weight.safetensors"
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

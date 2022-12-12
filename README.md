# Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning

<!-- #region -->
<p align="center">
<img  src="contents/alpha_scale.gif">
</p>
<!-- #endregion -->

> Using LoRA to fine tune on illustration dataset : $W = W_0 + \alpha \Delta W$, where $\alpha$ is the merging ratio. Above gif is scaling alpha from 0 to 1. Setting alpha to 0 is same as using the original model, and setting alpha to 1 is same as using the fully fine-tuned model.

<!-- #region -->
<p align="center">
<img  src="contents/lora_with_clip.jpg">
</p>
<!-- #endregion -->

> "game character bnha, wearing a red shirt, riding a donkey", with Overwatch-fine-tuned LoRA model, for both CLIP and Unet.

<!-- #region -->
<p align="center">
<img  src="contents/disney_lora.jpg">
</p>
<!-- #endregion -->

> "style of sks, baby lion", with disney-style LoRA model.

<!-- #region -->
<p align="center">
<img  src="contents/pop_art.jpg">
</p>
<!-- #endregion -->

> "style of sks, superman", with pop-art style LoRA model.

# Web Demo

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ysharma/Low-rank-Adaptation)

## Main Features

- Fine-tune Stable diffusion models twice as faster than dreambooth method, by Low-rank Adaptation
- Get insanely small end result, easy to share and download.
- Easy to use, compatible with diffusers
- Sometimes even better performance than full fine-tuning (but left as future work for extensive comparisons)
- Merge checkpoints by merging LoRA
- Fine-tune both CLIP & Unet to gain better results.

# UPDATES & Notes

- **You can now fine-tune text_encoder as well! Enabled with simple `--train_text_encoder`**
- **Converting to CKPT format for A1111's repo consumption!** (Thanks to [jachiam](https://github.com/jachiam)'s conversion script)
- Img2Img Examples added.
- Please use large learning rate! Around 1e-4 worked well for me, but certainly not around 1e-6 which will not be able to learn anything.

# Lengthy Introduction

Thanks to the generous work of Stability AI and Huggingface, so many people have enjoyed fine-tuning stable diffusion models to fit their needs and generate higher fidelity images. **However, the fine-tuning process is very slow, and it is not easy to find a good balance between the number of steps and the quality of the results.**

Also, the final results (fully fined-tuned model) is very large. Some people instead works with textual-inversion as an alternative for this. But clearly this is suboptimal: textual inversion only creates a small word-embedding, and the final image is not as good as a fully fine-tuned model.

Well, what's the alternative? In the domain of LLM, researchers have developed Efficient fine-tuning methods. LoRA, especially, tackles the very problem the community currently has: end users with Open-sourced stable-diffusion model want to try various other fine-tuned model that is created by the community, but the model is too large to download and use. LoRA instead attempts to fine-tune the "residual" of the model instead of the entire model: i.e., train the $\Delta W$ instead of $W$.

$$
W' = W + \Delta W
$$

Where we can further decompose $\Delta W$ into low-rank matrices : $\Delta W = A B^T $, where $A, \in \mathbb{R}^{n \times d}, B \in \mathbb{R}^{m \times d}, d << n$.
This is the key idea of LoRA. We can then fine-tune $A$ and $B$ instead of $W$. In the end, you get an insanely small model as $A$ and $B$ are much smaller than $W$.

Also, not all of the parameters need tuning: they found that often, $Q, K, V, O$ (i.e., attention layer) of the transformer model is enough to tune. (This is also the reason why the end result is so small). This repo will follow the same idea.

Enough of the lengthy introduction, let's get to the code.

# Installation

```bash
pip install git+https://github.com/cloneofsimo/lora.git
```

# Getting Started

## Fine-tuning Stable diffusion with LoRA.

Basic usage is as follows: prepare sets of $A, B$ matrices in an unet model, and fine-tune them.

```python
from lora_diffusion import inject_trainable_lora, extract_lora_up_downs

...

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="unet",
)
unet.requires_grad_(False)
unet_lora_params, train_names = inject_trainable_lora(unet)  # This will
# turn off all of the gradients of unet, except for the trainable LoRA params.
optimizer = optim.Adam(
    itertools.chain(*unet_lora_params, text_encoder.parameters()), lr=1e-4
)
```

An example of this can be found in `train_lora_dreambooth.py`. Run this example with

```bash
run_lora_db.sh
```

Another dreambooth example, with text_encoder training on can be run with:

```bash
run_lora_db_w_text.sh
```

## Loading, merging, and interpolating trained LORAs with CLIs.

We've seen that people have been merging different checkpoints with different ratios, and this seems to be very useful to the community. LoRA is extremely easy to merge.

By the nature of LoRA, one can interpolate between different fine-tuned models by adding different $A, B$ matrices.

Currently, LoRA cli has three options : merge full model with LoRA, merge LoRA with LoRA, or merge full model with LoRA and changes to `ckpt` format (original format)

```
SYNOPSIS
    lora_add PATH_1 PATH_2 OUTPUT_PATH <flags>

POSITIONAL ARGUMENTS
    PATH_1
        Type: str
    PATH_2
        Type: str
    OUTPUT_PATH
        Type: str

FLAGS
    --alpha
        Type: float
        Default: 0.5
    --mode
        Type: Literal['upl', 'lpl', 'upl', 'upl-ckpt-v2']
        Default: 'lpl'
    --with_text_lora
        Type: bool
        Default: False
```

### Merging full model with LoRA

```bash
$ lora_add --path_1 PATH_TO_DIFFUSER_FORMAT_MODEL --path_2 PATH_TO_LORA.PT --mode upl --alpha 1.0 --output_path OUTPUT_PATH
```

`path_1` can be both local path or huggingface model name. When adding LoRA to unet, alpha is the constant as below:

$$
W' = W + \alpha \Delta W
$$

So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can tune these values to your needs. This value can be even slightly greater than 1.0!

**Example**

```bash
$ lora_add --path_1 stabilityai/stable-diffusion-2-base --path_2 lora_illust.pt --mode upl --alpha 1.0 --output_path merged_model
```

### Mergigng Full model with LoRA and changing to original CKPT format

_TESTED WITH V2, V2.1 ONLY!_

Everything same as above, but with mode `upl-ckpt-v2` instead of `upl`.

```bash
$ lora_add --path_1 stabilityai/stable-diffusion-2-base --path_2 lora_illust.pt --mode upl-ckpt-v2 --alpha 1.2 --output_path merged_model.ckpt
```

### Merging LoRA with LoRA

```bash
$ lora_add --path_1 PATH_TO_LORA.PT --path_2 PATH_TO_LORA.PT --mode lpl --alpha 0.5 --output_path OUTPUT_PATH.PT
```

alpha is the ratio of the first model to the second model. i.e.,

$$
\Delta W = (\alpha A_1 + (1 - \alpha) A_2) (B_1 + (1 - \alpha) B_2)^T
$$

Set alpha to 0.5 to get the average of the two models. Set alpha close to 1.0 to get more effect of the first model, and set alpha close to 0.0 to get more effect of the second model.

**Example**

```bash
$ lora_add --path_1 lora_illust.pt --path_2 lora_pop.pt --alpha 0.3 --output_path lora_merged.pt
```

### More bash examples with Text Encoder Lora:

```bash
$ lora_add --path_1 stabilityai/stable-diffusion-2-base --path_2 lora_kiriko.pt --mode upl-ckpt-v2 --alpha 1.2 --with_text_lora --output_path merged_model.ckpt
```

: This will build a `merged_model.ckpt` with LoRA merged with $\alpha=1.2$ and text encoder LoRA.

### Making Text2Img Inference with trained LoRA

Checkout `scripts/run_inference.ipynb` for an example of how to make inference with LoRA.

### Making Img2Img Inference with LoRA

Checkout `scripts/run_img2img.ipynb` for an example of how to make inference with LoRA.

---

# Tips and Suggestions

### **How long should you train?**

Effect of fine tuning (both Unet + CLIP) can be seen in the following image, where each image is another 500 steps.
Trained with 9 images, with lr of `1e-4` for unet, and `5e-5` for CLIP. (You can adjust this with `--learning_rate=1e-4` and `--learning_rate_text=5e-5`)

<!-- #region -->
<p align="center">
<img  src="contents/lora_with_clip_4x4_training_progress.jpg">
</p>
<!-- #endregion -->

> "female game character bnha, in a steampunk city, 4K render, trending on artstation, masterpiece". Visualization notebook can be found at scripts/lora_training_process_visualized.ipynb

You can see that with 2500 steps, you already get somewhat good results.

### **What is a good learning rate for LoRA?**

People using dreambooth are used to using lr around `1e-6`, but this is way too small for training LoRAs. **I've tried using 1e-4, and it is OK**. I think these values should be more explored statistically.

### **What happens to Text Encoder LoRA and Unet LoRA?**

Let's see: the following is only using Unet LoRA:

<!-- #region -->
<p align="center">
<img  src="contents/lora_just_unet.jpg">
</p>
<!-- #endregion -->

And the following is only using Text Encoder LoRA:

<!-- #region -->
<p align="center">
<img  src="contents/lora_just_text_encoder.jpg">
</p>
<!-- #endregion -->

So they learnt different aspect of the dataset, but they are not mutually exclusive. You can use both of them to get better results, and tune them seperately to get even better results.

With LoRA Text Encoder, Unet, all the schedulers, guidance scale, negative prompt etc. etc., you have so much to play around with to get the best result you want. For example, with $\alpha_{unet} = 0.6$, $\alpha_{text} = 0.9$, you get a better result compared to $\alpha_{unet} = 1.0$, $\alpha_{text} = 1.0$ (default). Checkout below:

<!-- #region -->
<p align="center">
<img  src="contents/lora_some_tweaks.jpg">
</p>
<!-- #endregion -->

> Left with tuned $\alpha_{unet} = 0.6$, $\alpha_{text} = 0.9$, right with $\alpha_{unet} = 1.0$, $\alpha_{text} = 1.0$.

---

TODOS

- Make this more user friendly for non-programmers
- Make a better CLI
- Make a better documentation
- Kronecker product, like LoRA [https://arxiv.org/abs/2106.04647]
- Adaptor-guidance
- Time-aware fine-tuning.
- Test alpha scheduling. I think it will be meaningful.

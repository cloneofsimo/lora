# Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning

![name](contents/alpha_scale.mp4)

## Main Features

- Fine-tune Stable diffusion models twice as faster than dreambooth method, by Low-rank Adaptation
- Get insanely small end result, easy to share and download.
- Easy to use, compatible with diffusers
- Sometimes even better performance than full fine-tuning (but left as future work for extensive comparisons)
- Merge checkpoints by merging LORA

# Lengthy Introduction

Thanks to the generous work of Stability AI and Huggingface, so many people have enjoyed fine-tuning stable diffusion models to fit their needs and generate higher fidelity images. **However, the fine-tuning process is very slow, and it is not easy to find a good balance between the number of steps and the quality of the results.**

Also, the final results (fully fined-tuned model) is very large. Some people instead works with textual-inversion as an alternative for this. But clearly this is suboptimal: textual inversion only creates a small word-embedding, and the final image is not as good as a fully fine-tuned model.

Well, what's the alternative? In the domain of LLM, researchers have developed Efficient fine-tuning methods. LORA, especially, tackles the very problem the community currently has: end users with Open-sourced stable-diffusion model want to try various other fine-tuned model that is created by the community, but the model is too large to download and use. LORA instead attempts to fine-tune the "residual" of the model instead of the entire model: i.e., train the $\Delta W$ instead of $W$.

$$
W' = W + \Delta W
$$

Where we can further decompose $\Delta W$ into low-rank matrices : $\Delta W = A B^T $, where $A, \in \mathbb{R}^{n \times d}, B \in \mathbb{R}^{m \times d}, d << n$.
This is the key idea of LORA. We can then fine-tune $A$ and $B$ instead of $W$. In the end, you get an insanely small model as $A$ and $B$ are much smaller than $W$.

Also, not all of the parameters need tuning: they found that often, $Q, K, V, O$ (i.e., attention layer) of the transformer model is enough to tune. (This is also the reason why the end result is so small). This repo will follow the same idea.

Enough of the lengthy introduction, let's get to the code.

# Installation

```bash
pip install git+https://github.com/cloneofsimo/lora.git
```

# Getting Started

## Fine-tuning Stable diffusion with LORA.

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
# turn off all of the gradients of unet, except for the trainable LORA params.
optimizer = optim.Adam(
    itertools.chain(*unet_lora_params, text_encoder.parameters()), lr=1e-4
)
```

An example of this can be found in `train_lora_dreambooth.py`. Run this example with

```bash
run_lora_db.sh
```

## Loading, merging, and interpolating trained LORAs.

We've seen that people have been merging different checkpoints with different ratios, and this seems to be very useful to the community. LORA is extremely easy to merge.

By the nature of LORA, one can interpolate between different fine-tuned models by adding different $A, B$ matrices.

Currently, LORA cli has two options : merge unet with LORA, or merge LORA with LORA.

### Merging unet with LORA

```bash
$ lora_add --path_1 PATH_TO_DIFFUSER_FORMAT_MODEL --path_2 PATH_TO_LORA.PT --mode upl --alpha 1.0 --output_path OUTPUT_PATH
```

`path_1` can be both local path or huggingface model name. When adding LORA to unet, alpha is the constant as below:

$$
W' = W + \alpha \Delta W
$$

So, set alpha to 1.0 to fully add LORA. If the LORA seems to have too much effect (i.e., overfitted), set alpha to lower value. If the LORA seems to have too little effect, set alpha to higher than 1.0. You can tune these values to your needs.

**Example**

```bash
$ lora_add --path_1 stabilityai/stable-diffusion-2-base --path_2 lora_illust.pt --mode upl --alpha 1.0 --output_path merged_model
```

### Merging LORA with LORA

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

### Making Inference with trained LORA

Checkout `scripts/run_inference.ipynb` for an example of how to make inference with LORA.

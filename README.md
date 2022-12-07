# Low-rank Adapation for Fast Text-to-Image Diffusion Fine-tuning

Want to fine-tune a diffusion model, but irretated by the long training time + enormous end-result model? Annoyed that everyone creates their own checkpoints, and you have to literally download 100 different models just to find what's best for you? Want to quickly merge checkpoints, but without making another insane checkpoint model? _LORA is for you._

# Lengthy Introduction

Thanks to the generous work by Stability AI and Huggingface, so many people seems to enjoy fine-tuning stable diffusion models to fit their needs and generate higher fidelity images. **However, the fine-tuning process is very slow, and it is not easy to find a good balance between the number of steps and the quality of the results.**

Also, the final results (fully fined-tuned model) is indeed very large. Some people instead works with textual-inversion as alternative for this. But clearly this is suboptimal : textual inversion only creates a small word-embedding, and the final image is not as good as a fully fine-tuned model.

Well what's the alternative? In the domain of Large-language model, researhers have developed Efficient fine-tuning methods. LORA, especially, tackles the very problem the community currently has : end users with Open-sourced stable-diffusion model wants to try various other fine-tuned model that is created by the community, but the model is too large to download and use. LORA instead attempts to fine-tune "residual" of the model instead of the entire model : i.e., train the $\Delta W$ instead of $W$.

$$
W' = W + \Delta W
$$

Where we can further decompose $\Delta W$ into low-rank matrices : $\Delta W = A B^T $, where $A, \in \mathbb{R}^{n \times d}, B \in \mathbb{R}^{m \times d}, d << n$.
This is the key idea of LORA. We can then fine-tune $A$ and $B$ instead of $W$. At the end, you get insanely small model as $A$ and $B$ are much smaller than $W$.

Also, not all of parameters needs tuning : they found that often, $Q, K, V, O$ (i.e., attention layer) of the transformer model is enough to tune. (This is also the reason why the end result is so small). This repo will follow the same idea.

Enough of the lengthy introduction, let's get to the code.

# Installation

#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="./data_example_text"
export OUTPUT_DIR="./output_example_text"

accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="game character bnha" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-5 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000
#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data_example_text"
export OUTPUT_DIR="./output_example_lorpt"

accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=5000 \
  --placeholder_token="<krk>" \
  --learnable_property="object"\
  --initializer_token="woman" \
  --save_steps=500 \
  --unfreeze_lora_step=1500 \
  --stochastic_attribute="game character,3d render,4k,highres" # these attributes will be randomly appended to the prompts
  
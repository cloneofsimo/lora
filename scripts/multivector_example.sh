export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data_example"
export OUTPUT_DIR="./exps/output_example"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=3e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --placeholder_tokens="<krk1>|<krk2>|<krk3>|<krk4>" \
  --placeholder_token_at_data="<krk>|<krk1><krk2> character, wearing <krk3> hat, <krk4> cloth"\
  --use_template="object"\
  --initializer_tokens="<zero>|<zero>|<zero>|<zero>" \
  --save_steps=100 \
  --max_train_steps_ti=4000 \
  --max_train_steps_tuning=4000 \
  --perform_inversion=True \
  --weight_decay_ti=1. \
  --weight_decay_lora=0.1\
  --device="cuda:0"\
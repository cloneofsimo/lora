export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data_example_small"
export OUTPUT_DIR="./exps/output_example_enid_w_mask"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate_unet=3e-4 \
  --learning_rate_text=3e-4 \
  --learning_rate_ti=1e-3 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --placeholder_tokens="<s1>|<s2>|<s3>" \
  --placeholder_token_at_data="<s>|<s1><s2><s3>"\
  --initializer_tokens="girl|<rand>|<rand>" \
  --save_steps=100 \
  --max_train_steps_ti=500 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --use_template="object"\
  --weight_decay_ti=0.1 \
  --weight_decay_lora=0.001\
  --continue_inversion_lr=1e-4\
  --device="cuda:0"\
  --use_face_segmentation_condition\
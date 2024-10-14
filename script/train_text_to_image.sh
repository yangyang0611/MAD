MODEL_NAME="runwayml/stable-diffusion-v1-5"

accelerate launch --mixed_precision="fp16" sd_training/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir data/mtdataset/images \
  --train_json_file data/mt_text_anno.json \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=24000 \
  --checkpointing_steps=3000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="runs/sd_512_512" \

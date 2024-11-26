accelerate launch --num_cpu_threads_per_process 1 train_network.py \
    --pretrained_model_name_or_path=/home/user/lyq/sd-scripts-main/checkpoint/stable-diffusion-v1-5.safetensors\
    --dataset_config=/home/user/lyq/sd-scripts-main/config_lora.toml \
    --output_dir=/home/user/lyq/stable-diffusion-webui/models/Lora \
    --output_name=lyq4 \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_epochs=200 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --cache_latents \
    --gradient_checkpointing \
    --save_every_n_epochs=200 \
    --network_module=networks.lora
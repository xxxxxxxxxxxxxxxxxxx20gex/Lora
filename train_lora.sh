accelerate launch --num_cpu_threads_per_process 1 --num_processes 2\
    train_network.py \
    --pretrained_model_name_or_path=./checkpoint/stable-diffusion-v1-5.safetensors\
    --dataset_config=./config_lora.toml \
    --output_dir=./output/ \
    --output_name=lyq \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_epochs=20 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --cache_latents \
    --gradient_checkpointing \
    --save_every_n_epochs=200 \
    --network_module=networks.lora
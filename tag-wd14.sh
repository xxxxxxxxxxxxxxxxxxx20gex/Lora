python ./finetune/tag_images_by_wd14_tagger.py \
--onnx \
--thresh 0.8 \
--remove_underscore \
--always_first_tags "lyq"  \
--model_dir /home/user/lyq/sd-scripts-main/finetune/wd14_tagger_model111 \
--repo_id SmilingWolfwd-v1-4-convnext-tagger-v2 \
--batch_size 8 \
/home/user/lyq/sd-scripts-main/lyq_dataset/lyq_mask
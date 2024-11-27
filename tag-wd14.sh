python ./finetune/tag_images_by_wd14_tagger.py \
--onnx \
--thresh 0.8 \
--remove_underscore \
--batch_size 8 \
--model_dir ./finetune/SmilingWolfwd-swinv2-tagger-v3 \
/home/user/lyq/sd-scripts-main/lyq_dataset/lyq_mask
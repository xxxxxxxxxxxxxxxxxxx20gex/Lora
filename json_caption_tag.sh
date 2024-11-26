python finetune/make_captions.py --batch_size 8 /home/user/lyq/sd-scripts-main/lyq_dataset/lyq_mask
python /home/user/lyq/sd-scripts-main/finetune/merge_captions_to_metadata.py --full_path /home/user/lyq/sd-scripts-main/lyq_dataset/lyq_mask \
meta_cap_dataset3.json
python /home/user/lyq/sd-scripts-main/finetune/merge_dd_tags_to_metadata.py --full_path /home/user/lyq/sd-scripts-main/lyq_dataset/lyq_mask \
--in_json meta_cap_dataset3.json \
meta_cap_dd_dataset3.json
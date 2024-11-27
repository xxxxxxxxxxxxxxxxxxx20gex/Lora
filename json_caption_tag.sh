sh tage-wd14.sh
python finetune/make_captions.py --batch_size 8 /home/user/lyq/sd-scripts-main/lyq_dataset/lyq_mask
python ./finetune/merge_captions_to_metadata.py --full_path /home/user/lyq/sd-scripts-main/lyq_dataset/lyq_mask \
meta_caption2.json
python ./finetune/merge_dd_tags_to_metadata.py --full_path /home/user/lyq/sd-scripts-main/lyq_dataset/lyq_mask \
--in_json meta_captin2.json  meta_cap_dd2.json
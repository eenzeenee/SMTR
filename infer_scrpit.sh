#python inference.py  --gradient_clip_val 1.0 \
#        --max_epochs 3 --default_root_dir logs \
#        --model_path kobart_from_pretrained  \
#        --tokenizer_path emji_tokenizer \
#        --rap  --gpus -1 \
#        --train_file hiphop_data/train.csv --test_file hiphop_data/test.csv


python inference.py --model_path pretrained_dir/rap-kobart-model --tokenizer_path emji_tokenizer --rap
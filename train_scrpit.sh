python train.py  --gradient_clip_val 1.0 \
        --max_epochs 3 --default_root_dir logs \
        --model_path kobart_from_pretrained  \
        --tokenizer_path emji_tokenizer \
        --rhyme --gpus 1 \
        --train_file data/train_7_9.csv --test_file data/test_7_9.csv
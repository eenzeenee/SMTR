python train.py  --gradient_clip_val 1.0 \
        --max_epochs 1 --default_root_dir logs \
        --model_path kobart_from_pretrained  \
        --tokenizer_path emji_tokenizer \
        --rap --gpus 1 \
        --train_file hiphop_data/sample_train.csv --test_file hiphop_data/sample_test.csv
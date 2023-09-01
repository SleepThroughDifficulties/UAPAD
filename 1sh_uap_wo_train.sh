nohup python -u uap_wo_train.py \
    --dataset_name imdb \
    --task_name none \
    --ckpt_dir /20221110/saved_models/finetune/imdb/none/finetune_bert-base-uncased_imdb_lr2e-05_epochs3/epoch2/ \
    --max_seq_length 256 \
    --model_name bert-base-uncased \
    --perturb_num 10 \
    --attack_steps 300 \
    --class_bound 0.9 \
    --seed 42 \
    --cuda 0 > train_logs/train-imdb_test.log 2>&1 &
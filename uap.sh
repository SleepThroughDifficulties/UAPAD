nohup python uap.py \
    --task_name sst2\
    --model_name bert-base-uncased \
    --dataset_name glue \
    --epochs 3 \
    --perturb_num 10 \
    --attack_steps 300 \
    --class_bound 0.8 \
    --seed 42 > train_logs/QNLI_uap.log 2>&1 &
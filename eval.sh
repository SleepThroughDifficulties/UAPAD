nohup python eval.py \
    --task_name qnli\
    --model_name bert-base-uncased \
    --dataset_name glue \
    --epochs 3 \
    --seed 42 > train_logs/QNLI_eval.log 2>&1 &
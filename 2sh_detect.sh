DATASET_NAME="imdb"
ATTACK_NAME="textfooler"

nohup python -u adv_detect.py \
    --dataset_name $DATASET_NAME \
    --model_name /20221110/saved_models/finetune/imdb/none/finetune_bert-base-uncased_imdb_lr2e-05_epochs3/epoch2/ \
    --attack_name $ATTACK_NAME \
    --delta_weight 1.0 \
    --delta0_index 1 \
    --delta1_index 1 \
    --cuda_device 0 \
    --do_search \
    --scenario easy \
    --seed 42 > train_logs/detect-$DATASET_NAME-$ATTACK_NAME.log 2>&1 &
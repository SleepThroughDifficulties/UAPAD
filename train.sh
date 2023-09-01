nohup python train.py --task_name sst2 \
      --epochs 4 \
      --max_seq_length 128 \
      --bsz 32 \
      --lr 2e-5 \
      --weight_decay 1e-6 \
      --modeldir UAP_SST2 \
      --cuda 2 \
      --seed 42 > experiment_result/UAP_SST2_modeltrain.log 2>&1 &

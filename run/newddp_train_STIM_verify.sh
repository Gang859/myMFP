#!/bin/bash
NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7
MASTER_PORT=20213

cd src
shift
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./newddp_transformer_STIM_verify.py zrg \
    --exp_id model5_val4result \
    --gpus ${GPUS} \
    --gpus_offset 0 \
    --batch_size 512 \
    --num_epochs 1000 \
    --num_workers 64 \
    --weight_decay 0.01 \
    --model_type 4 \
    --ModelWithLoss_Type 1 \
    --lr 0.0005 \
    --specific_lr 0.0003 \
    --load_model /mnt/zhangrengang/workspace/myMFP/exp/zrg/model5_val4result/backup/800_epoch_model_last.pth
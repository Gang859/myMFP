#!/bin/bash
NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7
MASTER_PORT=20213

cd src
shift
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./newddp_transformer_STIM.py zrg \
    --exp_id model4_val4result \
    --gpus ${GPUS} \
    --gpus_offset 0 \
    --batch_size 16 \
    --num_epochs 800 \
    --num_workers 64 \
    --weight_decay 0.01 \
    --model_type 4 \
    --lr 0.0003 \
    --load_model /mnt/zhangrengang/workspace/myMFP/exp/zrg/model4_val4result/model_last.pth

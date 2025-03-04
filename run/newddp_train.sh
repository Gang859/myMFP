#!/bin/bash
NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7
MASTER_PORT=20212

cd src
shift
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./newddp_transformer.py zrg \
    --exp_id model2 \
    --gpus ${GPUS} \
    --gpus_offset 0 \
    --batch_size 8 \
    --num_epochs 160 \
    --num_workers 64 \
    --weight_decay 0.001 \
    --model_type 2 \
    --lr 0.0001 \
    --load_model /mnt/zhangrengang/workspace/myMFP/exp/zrg/model2/model_last.pth
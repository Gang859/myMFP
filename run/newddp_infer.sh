#!/bin/bash
NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7
MASTER_PORT=20213

cd src
shift
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./newddp_transformer_with_infer.py zrg \
    --exp_id infer \
    --gpus ${GPUS} \
    --gpus_offset 0 \
    --batch_size 32 \
    --num_epochs 60 \
    --num_workers 32 \
    --load_model /mnt/zhangrengang/workspace/myMFP/exp/zrg/model2/backup/model_best.pth
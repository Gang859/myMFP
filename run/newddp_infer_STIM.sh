#!/bin/bash
NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7
MASTER_PORT=20242

cd src
shift
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./newddp_transformer_with_infer_STIM.py zrg \
    --exp_id model4_val4result_infer \
    --gpus ${GPUS} \
    --gpus_offset 0 \
    --batch_size 16 \
    --num_epochs 1 \
    --num_workers 16 \
    --model_type 4 \
    --load_model /mnt/zhangrengang/workspace/myMFP/exp/zrg/model4_val4result/backup/421_epoch_model_last.pth
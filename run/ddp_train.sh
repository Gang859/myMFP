CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    /mnt/zhangrengang/workspace/myMFP/src/ddp_transformer.py
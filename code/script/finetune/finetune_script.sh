#!/bin/bash


DATASET_NAME="your_dataset_name"  # 'qqp','sst-2'
DATASET_PATH="your_data_path" 
output_path="your_output_path"
pretrained_weight="your_pretrained_weight_path"


python -u train_sst2.py \
    --cuda \
    --data ${DATASET_PATH} \
    --dataset ${DATASET_NAME} \
    --n_layer 8 \
    --d_model 352 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 512 \
    --dropout 0.1 \
    --dropatt 0.1 \
    --optim adam \
    --lr 1e-4 \
    --warmup_step 0 \
    --max_step 4000 \
    --eval-interval 500 \
    --log-interval 100 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 16 \
    --multi_gpu \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --gate_name GlobalRouterGate \
    --global_router_heads 8 \
    --global_router_dropout 0.1 \
    --global_router_d_kv 32 \
    --global_router_k 3 \
    --moe_index 0,1,2,3 \
    --load_balance 0.01 \
    --work_dir  ${output_path}\
    --pretrained_weight  ${pretrained_weight}

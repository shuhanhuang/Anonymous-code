#!/bin/bash
DATASET_PATH="your_data_path"
output_path="your_output_path"
your_dataset_name="your_dataset_name"

python -u train_global.py \
    --cuda \
    --data $DATASET_PATH \
    --dataset $your_dataset_name \
    --n_layer 8 \
    --d_model 352 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 512 \
    --dropout 0.1 \
    --dropatt 0.1 \
    --optim adam \
    --lr 0.0007 \
    --warmup_step 4000 \
    --max_step 80000 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 48 \
    --multi_gpu \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --gate_name GlobalRouterGate \
    --global_router_heads 8 \
    --global_router_dropout 0.1 \
    --global_router_d_kv 64 \
    --global_router_k 3 \
    --moe_index 0,1,2,3 \
    --load_balance 0.01 \
    --freeze_gate \
    --dynamic_overall_steps 80000 \
    --work_dir $output_path
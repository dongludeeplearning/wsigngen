#!/bin/bash

python -m train.train_mdm  \
    --save_dir save/wlasl30_ckp_05.11_t+d \
    --dataset wlasl30 \
    --cond_mask_prob 0 \
    --lambda_rcxyz 0 \
    --lambda_vel 1 \
    --lambda_fc 1 \
    --batch_size 128 \
    --cond_mode  t+d \
    --overwrite 




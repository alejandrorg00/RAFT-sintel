#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-sintel-flyvis-split-rgb --stage sintel_flyvis_split_rgb --validation sintel_flyvis_split_rgb ...
python -u train.py --name raft-sintel-flyvis-split-lum-bs8 --stage sintel_flyvis_split_lum --validation sintel_flyvis_split_lum --gpus 0 --num_steps 100000 --batch_size 8 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma 0.85 --mixed_precision
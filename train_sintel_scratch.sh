#!/bin/bash
mkdir -p checkpoints

python -u train.py --name raft-sintel-scratch --stage sintel_scratch --validation sintel --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma 0.85 --mixed_precision
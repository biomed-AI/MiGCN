#!/bin/bash

emb_dim=200
hid_dim=200
layer_size="[200,200]"
batch_size=512
learning_rate=0.005
dropout=0.3
reg=0.001

DATA="./OMIM/bigraph/cross1"
num_epochs=1000
seed=2022
gamma=0.1
alpha=3
beta=0.1
GPU=-1
type=3
neg_num=5
k=0.5
mi_kind="js"
pretrain=./OMIM


python main.py \
    -dataset $DATA \
    -emb_dim $emb_dim \
    -hid_dim $hid_dim \
    -layer_size $layer_size \
    -batch_size $batch_size \
    -neg_num $neg_num \
    -num_epochs $num_epochs \
    -learning_rate $learning_rate \
    -dropout $dropout \
    -reg $reg \
    -gamma $gamma \
    -k $k \
    -mi_kind $mi_kind \
    -alpha $alpha \
    -beta $beta \
    -seed $seed \
    -type $type \
    -gpu_id $GPU
    # -pretrain $pretrain \

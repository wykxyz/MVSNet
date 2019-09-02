#!/bin/bash
n=1
part=AD
name=v2_gpu1_3DCNNs
srun --mpi=pmi2 --gres=gpu:${n}  \
        -p $part -n1 \
        --ntasks-per-node=1 \
        -J $name -K \
        python2 -u test.py \
        --regularization='GRU_WGATE' \
        --model_dir=../model/0808/4_GRU_WGATE_v3_d128_square_channel_sum3 \
        --ckpt_step=160000 \
        --dense_folder=/mnt/lustre/share/yihongwei/dataset/mvs-test/family \
        --/mnt/lustre/share/yihongwei/dataset &

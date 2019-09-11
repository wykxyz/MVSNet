#!/bin/bash
n=1
part=AD
name=evaluate
#        --ckpt_step=160000 \
test_list=(horse fransic lighthouse m60 panther playground train)
#test_list=(family)
for data in ${test_list[@]}
do
echo 'process '+$data
srun --mpi=pmi2 --gres=gpu:${n}  \
        -p $part -n1 \
        --ntasks-per-node=1 \
        -J $name -K \
        python2 -u test.py \
        --regularization='GRU' \
        --model_dir=/mnt/lustre/share/yihongwei/workspace/model/0805_4_GRU \
        --dense_folder=/mnt/lustre/share/yihongwei/dataset/mvs-test/${data} \
        --ckpt_step=325000 \
        --max_w=1920 \
        --max_h=1040 \
        --max_d=256 \
        --interval_scale=0.8 \
        --upsampling=True &
done

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
        --regularization='GRU_NONLOCALVIEWNUM' \
        --model_dir=/mnt/lustre/zhouhui/yihongwei/workspace/MVSNet/model/0905_4_GRU_NONLOCALVIEWNUM_v3_d128_i1.59 \
        --dense_folder=/mnt/lustre/share/yihongwei/dataset/mvs-test/${data} \
        --ckpt_step=135000 \
        --max_w=1920 \
        --max_h=1040 \
        --max_d=256 \
        --interval_scale=0.8 \
        --upsampling=True &
done

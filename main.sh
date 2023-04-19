#!/usr/bin/env bash
mode=$1
dataset=$2
lr=$3
batchsize=$4
modeltype=$5
epochs=$6
tareps=$7
for RUN in 1 2 3
do
    python3 main.py --mode $mode --dataset $dataset --lr $lr --batch_size $batchsize --model_type $modeltype --epochs $epochs --tar_eps $tareps --seed $RUN
done

exit $?
#!/bin/bash

base_dir="./data_v2/"
mkdir -p ${base_dir}/train/ALL

cat ${base_dir}/train/LCQMC/train ${base_dir}/train/BQ/train ${base_dir}/train/OPPO/train > ${base_dir}/train/ALL/train
cat ${base_dir}/train/LCQMC/dev ${base_dir}/train/BQ/dev ${base_dir}/train/OPPO/dev > ${base_dir}/train/ALL/dev
cat ${base_dir}/test/test_A | awk -F"\t" 'BEGIN{OFS="\t"}{print $4,$5}' > ${base_dir}/test/public_test_A
cat ${base_dir}/test/test_B | awk -F"\t" 'BEGIN{OFS="\t"}{print $4,$5}' > ${base_dir}/test/public_test_B

echo "Merge LCQMC、BQ、OPPO train_set as final train_set."
echo "Merge LCQMC、BQ、OPPO dev_set as final dev_set."

#!/usr/bin/env bash

# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37
# 执行目录：需说明
cd .

# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
pip install sentencepiece  # 安装 sentencepiece

# 2 拷贝该模型需要数据、预训练模型
# 这一步无需操作，数据和模型会自动下载

# 3 批量运行（如不方便批量，1，2需放到单个模型中）
model_mode_list=(xlnet-base-cased)
fp_item_list=(fp32)
bs_item_list=(16 32 64 128)
for model_mode in ${model_mode_list[@]}; do
    for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_item_list[@]}; do
              echo "index is speed, 1gpus, begin, ${model_name}"
              run_mode=sp
              CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 1500 ${model_mode}     #  (5min)
              sleep 60
              echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
              run_mode=mp
              CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 1500 ${model_mode}
              sleep 60
          done
    done
done

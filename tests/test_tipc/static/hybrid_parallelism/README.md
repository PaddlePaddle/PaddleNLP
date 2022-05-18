# Benchmark for static mode gpt-3
静态图模式GPT-3 benchmark测试脚本说明

.
├── README.md
├── prepare.sh    # 训练准备脚本，完成数据下载和相关依赖安装
├── run_all.sh    # 批量模型执行脚本
└── run_benchmark.sh    # 单模型运行脚本，可完成指定模型的测试方案

# Docker 运行环境
docker image: registry.baidu.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.0-cudnn8
paddle = 2.3
python = 3.7
# 运行benchmark测试
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/examples/language_model/gpt-3/static/
bash benchmark/prepare.sh
# 运行指定模型
Usage：bash run_benchmark.sh ${dp_degree} ${pp_degree} ${pp_degree} ${use_amp} ${global_batch_size} ${micro_batch_size}

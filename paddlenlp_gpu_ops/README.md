# paddlenlp_gpu_ops
paddlenlp_gpu_ops 是一个专为 PaddleNLP 设计的 GPU 算子库，它囊括了一系列常用的自然语言处理（NLP）算子，并提供了 CUDA 和 Triton 两种实现方式，以充分利用 GPU 的强大计算能力。

目前支持：
- mamba1 && mamba2 算子
- fast_ln && fused_ln 算子
- ml-cross-entropy 算子
- inf_cl 算子

# 安装指南

## 编译 cuda 算子
```bash
cd csrc
rm -rf build dist *.egg-info  # 清理之前的构建文件和目录
python setup.py build  # 开始编译
```

## 打包 wheel 包
完成 CUDA 算子的编译后，接下来打包成 Wheel 包以便安装：
```bash
python setup.py bdist_wheel
```

## 安装 wheel 包
使用 pip 命令安装刚刚打包好的 Wheel 包：
```bash
pip install dist/*.whl
```

# Test
```bash
pytest -v tests/cuda  # 测试 CUDA 算子
pytest -v tests/triton  # 测试 Triton 算子
```

通过上述步骤，您将能够顺利安装并测试 `paddlenlp_gpu_ops` 库，享受 GPU 加速带来的高效 NLP 算子体验。

# 参考资料
- https://github.com/state-spaces/mamba
- https://github.com/Dao-AILab/causal-conv1d
- https://github.com/apple/ml-cross-entropy
- https://github.com/DAMO-NLP-SG/Inf-CLIP

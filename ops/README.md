# PaddleNLP Kernel 库
> paddlenlp-kernel 是一个专为 PaddleNLP 量身打造的 GPU 算子库，它集成了一系列常用的自然语言处理（NLP）算子，并提供了 CUDA 和 Triton 两种高效的实现方式，旨在充分利用 GPU 的卓越计算能力，为 NLP 任务加速。

当前支持的算子包括：
- mamba1 和 mamba2 算子
- fast_ln 和 fused_ln 算子
- ml-cross-entropy 算子
- inf_cl 算子

# 安装指南

## 编译 cuda 算子
要编译 `CUDA` 算子，请执行以下命令：
```bash
cd csrc
rm -rf build dist *.egg-info  # 清除旧的构建文件和目录
python setup.py build  # 开始新的编译过程
```

## 打包 wheel
编译完成后，您可以将 `CUDA` 算子打包成 `Wheel` 包以便安装：
```bash
python setup.py bdist_wheel
```

## 安装 wheel
使用 pip 命令安装刚刚生成的 Wheel 包：
```bash
pip install dist/*.whl
```

## 使用 paddlenlp_kernel 库
以下是如何在代码中使用 `CUDA` 和 `Triton` 算子的示例：
```python
# 导入并使用 CUDA 算子
from paddlenlp_kernel.cuda.selective_scan import selective_scan_fn
xxx = selective_scan_fn(xxx)

# 导入并使用 Triton 算子
from paddlenlp_kernel.triton.inf_cl import cal_flash_loss
xxx = cal_flash_loss(xxx)
```

# 测试

要测试 `CUDA` 和 `Triton` 算子，请分别运行以下命令：
```bash
pytest -v tests/cuda  # 测试 CUDA 算子
pytest -v tests/triton  # 测试 Triton 算子
```

通过上述步骤，您将能够顺利安装并测试 `paddlenlp_kernel` 库，享受 GPU 加速带来的高效 NLP 算子体验。

# 注意

推荐用户使用以下版本的库：
- paddlepaddle-gpu >= 3.0.0b2
- triton >= 3.0.0

由于 `Triton` 库原本依赖于 `PyTorch`，为了方便 `Paddle` 用户使用 `Triton`，您可以按照以下步骤替换 `Triton` 库的部分源码，使其与 `Paddle` 兼容：

```bash
python -m pip install git+https://github.com/zhoutianzi666/UseTritonInPaddle.git
# 只需执行一次以下命令，之后即可在任意终端中使用 Triton，无需重复执行
python -c "import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()"
```

# 参考资料
- https://github.com/state-spaces/mamba
- https://github.com/Dao-AILab/causal-conv1d
- https://github.com/apple/ml-cross-entropy
- https://github.com/DAMO-NLP-SG/Inf-CLIP

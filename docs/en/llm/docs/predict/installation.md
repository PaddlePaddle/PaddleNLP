# High-Performance Inference Operator Installation

Clone the code locally:

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
export PYTHONPATH=/path/to/PaddleNLP:$PYTHONPATH
```

PaddleNLP provides high-performance custom operators for Transformer series models to boost inference and decoding performance. Install the custom operator library first:

```shell
# Install custom operators for GPU
cd PaddleNLP/csrc && python setup_cuda.py install
# Install custom operators for XPU
cd PaddleNLP/csrc/xpu/src && sh cmake_build.sh
# Install custom operators for DCU
cd PaddleNLP/csrc && python setup_hip.py install
# Install custom operators for SDAA
cd PaddleNLP/csrc/sdaa && python setup_sdaa.py install
```

Install Triton dependencies:

```shell
pip install triton  # Recommended version 3.2.0

python -m pip install git+https://github.com/zhoutianzi666/UseTritonInPaddle.git

# Only need to execute this command once. No need to repeat in future sessions
python -c "import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()"
```

Navigate to the running directory to start:

```shell
cd PaddleNLP/llm
```

Large Model Inference Tutorials:

- [llama](./llama.md)
- [qwen](./qwen.md)
- [deepseek](./deepseek.md)
- [mixtral](./mixtral.md)

For Optimal Inference Performance:

- [Best Practices](./best_practices.md)

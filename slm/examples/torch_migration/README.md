# BERT-SST2-Prod
Reproduction process of BERT on SST2 dataset

# 安装说明

* 下载代码库

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration
```

* 进入文件夹，安装 requirements

```shell
pip install -r requirements.txt
```

* 安装 PaddlePaddle 与 PyTorch

```shell
# CPU版本的PaddlePaddle
pip install paddlepaddle==2.2.0 -i https://mirror.baidu.com/pypi/simple
# 如果希望安装GPU版本的PaddlePaddle，可以使用下面的命令
# pip install paddlepaddle-gpu==2.2.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# 安装PyTorch
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**注意**: 本项目依赖于 paddlepaddle-2.2.0版本，安装时需要注意。

* 验证 PaddlePaddle 是否安装成功

运行 python，输入下面的命令。

```shell
import paddle
paddle.utils.run_check()
print(paddle.__version__)
```

如果输出下面的内容，则说明 PaddlePaddle 安装成功。

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
2.2.0
```


* 验证 PyTorch 是否安装成功

运行 python，输入下面的命令，如果可以正常输出，则说明 torch 安装成功。

```shell
import torch
print(torch.__version__)
# 如果安装的是cpu版本，可以按照下面的命令确认torch是否安装成功
# 期望输出为 tensor([1.])
print(torch.Tensor([1.0]))
# 如果安装的是gpu版本，可以按照下面的命令确认torch是否安装成功
# 期望输出为 tensor([1.], device='cuda:0')
print(torch.Tensor([1.0]).cuda())
```

# FasterERNIE实现文本分类

近几年NLP Transformer类的模型发展迅速，各个NLP的基础技术和核心应用的核心技术基本上都被Transformer类的核心技术所替换。
学术上，目前各个NLP任务领域的SOTA效果基本都是由Transformer类刷新，但在落地应用上还面临上线困难的问题。
Transformer类文本预处理部分，主要存在以下两个因素影响Transformer训推一体的部署：

* 文本预处理部分复杂，不具备训推一体的部署体验，C++端需要重新开发，成本高。
  训练侧多为Python实现，C++部署时需要重新进行迁移与对齐，目前业界缺少标准高效的C++参考实现
* 文本预处理效率Python实现与C++实现存在数量级上的差距，对产业实践场景有较大的价值。
  在服务部署时Tokenizer的性能也是推动NLP相关模型的一个性能瓶颈，尤其是小型化模型部署如ERNIE-Tiny等，文本预处理耗时占总体预测时间高达30%。


基于以上两点原因，我们将Transformer类文本预处理部分内置成Paddle底层算子——FasterTokenizer。
FasterTokenizer底层为C++实现，同时提供了python接口调用。其可以将文本转化为模型数值化输入。
同时，用户可以将其导出为模型的一部分，直接用于部署推理。从而实现Transformer训推一体。

为了更好地实现训推一体化，PaddleNLP 2.2版本将文本预处理FasterTokenizer内置到ERNIE模型内形成FasterERNIE模型。

以下示例展示FasterERNIE用于序列标注任务。

## 环境依赖

* paddlepaddle >= 2.2.1

* paddlenlp >= 2.2

## 代码结构说明

以下是本项目主要代码结构及说明：

```text
token_cls/
├── cpp_deploy # cpp静态图推理
│   ├── CMakeLists.txt
│   ├── compile.sh # 编译脚本
│   ├── lib # 第三方依赖
│   │   ├── CMakeLists.txt
│   │   └── paddle_inference # 预测库lib
│   ├── run.sh # 执行脚本
│   └── token_cls_infer.cc # CPP源代码
├── export_model.py # 导出静态图模型
├── predict.py # 动态预测
├── python_deploy.py # python静态图推理
├── README.md # 文档说明
└── train.py # 动态图训练
```

## 训练

我们以中文序列标注公开数据集MSRA-NER为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在测试集（test.tsv）验证

```shell
export CUDA_VISIBLE_DEVICES=0
python train.py --device gpu --save_dir checkpoints/ --batch_size 32 --max_seq_length 128
```

可支持配置的参数：

* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `max_seq_length`：可选，ERNIE模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.00。
* `epochs`: 训练轮次，默认为3。
* `warmup_proportion`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.1。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。

程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── model_100
│   ├── model_config.json
│   ├── model_state.pdparams
│   └── vocab.txt
└── ...
```

**NOTE:**
* 如需恢复模型训练，则可以设置`init_from_ckpt`， 如`init_from_ckpt=checkpoints/model_100/model_state.pdparams`。
* 使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见export_model.py。静态图参数保存在`output_path`指定路径中。
  运行方式：

```shell
python export_model.py --save_dir=./checkpoints/model_2000/ --output_path=./export
```
其中`save_dir`是指动态图训练保存的参数路径，`output_path`是指静态图参数导出路径。


## 预测

启动预测：
```shell
export CUDA_VISIBLE_DEVICES=0
python predict.py --device gpu --save_dir checkpoints/model_2000/
```


## 部署推理

### Python

导出模型之后，可以用于部署，python_predict.py文件提供了Python部署预测示例。运行方式：

```shell
export CUDA_VISIBLE_DEVICES=0
python python_deploy.py --model_dir export/ --batch_size 1
```

### C++

同时，我们还提供了C++部署推理示例脚本。

首先需要从[官网](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html)下载2.2 版本以上paddle inference lib。
解压至cpp_deploy/lib文件目录下, 如
```
cpp_deploy/lib
├── CMakeLists.txt
└── paddle_inference
    ├── CMakeCache.txt
    ├── paddle
    │   ├── include                                    C++ 预测库头文件目录
    │   │   ├── crypto
    │   │   ├── internal
    │   │   ├── paddle_analysis_config.h
    │   │   ├── paddle_api.h
    │   │   ├── paddle_infer_declare.h
    │   │   ├── paddle_inference_api.h                 C++ 预测库头文件
    │   │   ├── paddle_mkldnn_quantizer_config.h
    │   │   └── paddle_pass_builder.h
    │   └── lib
    │       ├── libpaddle_inference.a                  C++ 静态预测库文件
    │       └── libpaddle_inference.so                 C++ 动态预测库文件
    ├── third_party
    │   ├── install                                    第三方链接库和头文件
    │   │   ├── cryptopp
    │   │   ├── gflags
    │   │   ├── glog
    │   │   ├── mkldnn
    │   │   ├── mklml
    │   │   ├── protobuf
    │   │   ├── utf8proc
    │   │   └── xxhash
    │   └── threadpool
    │       └── ThreadPool.h
    └── version.txt
```


- 首先，编译C++源码，生成可执行文件。
注意替换CUDNN_LIB、CUDA_LIB指定路径。

```shell
mkdir -p build
cd build

# same with the token_cls_infer.cc
PROJECT_NAME=token_cls_infer

WITH_MKL=ON
WITH_GPU=ON

LIB_DIR=$PWD/lib/paddle_inference
CUDNN_LIB=/path/to/cudnn/lib/
CUDA_LIB=/path/to/cuda/lib/

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DPROJECT_NAME=${PROJECT_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB}

make -j
```

- 之后，运行可执行文件token_cls_infer，即可完成推理。


```shell
token_cls_infer --model_file ../../export/inference.pdmodel --params_file ../../export/inference.pdiparams
```

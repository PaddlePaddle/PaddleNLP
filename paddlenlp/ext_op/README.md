# 自定义OP编译使用

## 子目录结构

```text
.
├── sample/                 # 基于 Transformer 机器翻译使用样例（beam search）
├── src/                    # 自定义 OP C++ CUDA 代码
└── transformer/            # Python API 封装脚本
```

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.0.1 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1 或是更新的版本（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/v3.1#setup) 使用必要的环境

## 快速开始

### 编译自定义OP

自定义 OP 需要将实现的 C++、CUDA 代码编译成动态库，我们提供对应的 CMakeLists.txt ，可以参考使用如下的方式完成编译。

#### 克隆 PaddleNLP

首先，因为需要基于当前环境重新编译，当前的 paddlenlp 的 python 包里面并不包含 Faster Transformer 相关 lib，需要克隆一个 PaddleNLP，并重新编译:
``` sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

其次，配置环境变量，让我们可以使用当前 clone 的 paddlenlp，并进入到自定义 OP 的路径，准备后续的编译操作：

``` sh
export PYTHONPATH=$PWD/PaddleNLP/:$PYTHONPATH
cd PaddleNLP/paddlenlp/ext_op/
```

#### 编译

编译之前，请确保安装的 PaddlePaddle 的版本需要大于 2.0.1，并且正常可用。

编译自定义 OP 可以参照一下步骤：

``` sh
mkdir build
cd build/
cmake .. -DSM=xx -DCMAKE_BUILD_TYPE=Release
make -j
```

注意：`xx` 是指的所用 GPU 的 compute capability。举例来说，可以将之指定为 70(V100) 或是 75(T4)。

最终，编译会在 `./build/lib/` 路径下，产出 `libdecoding_op.so`，即需要的 Faster Transformer decoding 执行的库。

#### 使用

编写 python 脚本的时候，调用 `FasterTransformer` API 并传入 `libdecoding_op.so` 的位置即可实现将 Faster Transformer 用于当前的预测。

举例如下：

``` python
from paddlenlp.ext_op import FasterTransformer

transformer = FasterTransformer(
    src_vocab_size=args.src_vocab_size,
    trg_vocab_size=args.trg_vocab_size,
    max_length=args.max_length + 1,
    n_layer=args.n_layer,
    n_head=args.n_head,
    d_model=args.d_model,
    d_inner_hid=args.d_inner_hid,
    dropout=args.dropout,
    weight_sharing=args.weight_sharing,
    bos_id=args.bos_idx,
    eos_id=args.eos_idx,
    beam_size=args.beam_size,
    max_out_len=args.max_out_len,
    decoding_lib=args.decoding_lib,
    use_fp16_decoding=args.use_fp16_decoding)
```

更详细的例子可以参考 `./sample/decoding_sample.py`，我们提供了更详细用例。

#### 模型推断

#### 使用动态图预测(使用 float32 decoding 预测)

以英德翻译数据为例，模型训练完成后可以执行以下命令对指定文件中的文本进行翻译：

``` sh
# setting visible devices for prediction
export CUDA_VISIBLE_DEVICES=0
python sample/decoding_sample.py --config ./sample/config/transformer.base.yaml --decoding-lib ./build/lib/libdecoding_op.so
```

其中，`--config` 选项用于指明配置文件的位置，而 `--decoding-lib` 选项用于指明编译好的 Faster Transformer decoding lib 的位置。

翻译结果会输出到 `output_file` 指定的文件。执行预测时需要设置 `init_from_params` 来给出模型所在目录，更多参数的使用可以在 `./sample/config/transformer.base.yaml` 文件中查阅注释说明并进行更改设置。如果执行不提供 `--config` 选项，程序将默认使用 base model 的配置。

需要注意的是，目前预测仅实现了单卡的预测，原因在于，翻译后面需要的模型评估依赖于预测结果写入文件顺序，多卡情况下，目前暂未支持将结果按照指定顺序写入文件。

#### 使用动态图预测(使用 float16 decoding 预测)

float16 与 float32 预测的基本流程相同，不过在使用 float16 的 decoding 进行预测的时候，需要在配置文件中修改 `use_fp16_decoding` 配置为 `True`。后按照与之前相同的方式执行即可。具体执行方式如下：

``` sh
# setting visible devices for prediction
export CUDA_VISIBLE_DEVICES=0
python sample/decoding_sample.py --config ./sample/config/transformer.base.yaml --decoding-lib ./build/lib/libdecoding_op.so
```

其中，`--config` 选项用于指明配置文件的位置，而 `--decoding-lib` 选项用于指明编译好的 Faster Transformer decoding lib 的位置。

翻译结果会输出到 `output_file` 指定的文件。执行预测时需要设置 `init_from_params` 来给出模型所在目录，更多参数的使用可以在 `./sample/config/transformer.base.yaml` 文件中查阅注释说明并进行更改设置。如果执行不提供 `--config` 选项，程序将默认使用 base model 的配置。

需要注意的是，目前预测仅实现了单卡的预测，原因在于，翻译后面需要的模型评估依赖于预测结果写入文件顺序，多卡情况下，目前暂未支持将结果按照指定顺序写入文件。

## 模型评估

预测结果中每行输出是对应行输入的得分最高的翻译，对于使用 BPE 的数据，预测出的翻译结果也将是 BPE 表示的数据，要还原成原始的数据（这里指 tokenize 后的数据）才能进行正确的评估。评估过程具体如下（BLEU 是翻译任务常用的自动评估方法指标）：

``` sh
# 还原 predict.txt 中的预测结果为 tokenize 后的数据
sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt
# 若无 BLEU 评估工具，需先进行下载
git clone https://github.com/moses-smt/mosesdecoder.git
# 以英德翻译 newstest2014 测试数据为例
perl mosesdecoder/scripts/generic/multi-bleu.perl ~/.paddlenlp/datasets/machine_translation/WMT14ende/WMT14.en-de/wmt14_ende_data/newstest2014.tok.de < predict.tok.txt
```

执行上述操作之后，可以看到类似如下的结果，此处结果是 base model 在 newstest2014 上的 BLEU 结果：
```
BLEU = 26.86, 58.3/32.5/20.5/13.4 (BP=1.000, ratio=1.013, hyp_len=65326, ref_len=64506)
```

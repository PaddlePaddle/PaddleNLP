# FasterTransformer 预测

在这里我们集成了 NVIDIA [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1) 用于预测加速。同时集成了 FasterTransformer float32 以及 float16 预测。以下是使用 FasterTransformer 的说明。

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.1.0 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1 或 10.2（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup) 使用必要的环境
* 环境依赖
  - attrdict
  - pyyaml
  ```shell
  pip install attrdict pyyaml
  ```

## 快速开始

我们实现了基于 FasterTransformer 的自定义 op 的接入，用于加速当前机器翻译 example 在 GPU 上的预测性能。

## 使用 FasterTransformer 完成预测

编写 python 脚本的时候，调用 [`FasterTransformer` API](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.ops.faster_transformer.transformer.faster_transformer.html#paddlenlp.ops.faster_transformer.transformer.faster_transformer.FasterTransformer) 即可实现 Transformer 模型的高性能预测。

举例如下：

``` python
from paddlenlp.ops import FasterTransformer

transformer = FasterTransformer(
    src_vocab_size=args.src_vocab_size,
    trg_vocab_size=args.trg_vocab_size,
    max_length=args.max_length + 1,
    num_encoder_layers=args.n_layer,
    num_decoder_layers=args.n_layer,
    n_head=args.n_head,
    d_model=args.d_model,
    d_inner_hid=args.d_inner_hid,
    dropout=args.dropout,
    weight_sharing=args.weight_sharing,
    bos_id=args.bos_idx,
    eos_id=args.eos_idx,
    decoding_strategy=args.decoding_strategy,
    beam_size=args.beam_size,
    topk=args.topk,
    topp=args.topp,
    max_out_len=args.max_out_len,
    use_fp16_decoding=args.use_fp16_decoding)
```

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考 [文本生成高性能加速](../../../../paddlenlp/ops/README.md)。编译好后，使用 `FasterTransformer(decoding_lib="/path/to/lib", ...)` 可以完成导入。

更详细的例子可以参考 `encoder_decoding_predict.py`，我们提供了更详细用例。


#### 数据准备

公开数据集：WMT 翻译大赛是机器翻译领域最具权威的国际评测大赛，其中英德翻译任务提供了一个中等规模的数据集，这个数据集是较多论文中使用的数据集，也是 Transformer 论文中用到的一个数据集。我们也将[WMT'14 EN-DE 数据集](http://www.statmt.org/wmt14/translation-task.html)作为示例提供。

同时，我们提供了一份已经处理好的数据集，可以编写如下代码，对应的数据集将会自动下载并且解压到 `~/.paddlenlp/datasets/WMT14ende/`。

``` python
datasets = load_dataset('wmt14ende', splits=('test'))
```


#### 模型推断

使用模型推断前提是需要指定一个合适的 checkpoint，需要在对应的 `../configs/transformer.base.yaml` 中修改对应的模型载入的路径参数 `init_from_params`。

我们提供一个已经训练好的动态图的 base model 的 checkpoint 以供使用，可以通过[transformer-base-wmt_ende_bpe](https://bj.bcebos.com/paddlenlp/models/transformers/transformer/transformer-base-wmt_ende_bpe.tar.gz)下载。

``` sh
wget https://bj.bcebos.com/paddlenlp/models/transformers/transformer/transformer-base-wmt_ende_bpe.tar.gz
tar -zxf transformer-base-wmt_ende_bpe.tar.gz
```

然后，需要修改对应的 `../configs/transformer.base.yaml` 配置文件中的 `init_from_params` 的值为 `./base_trained_models/step_final/`。

#### 使用动态图预测(使用 float32 decoding 预测)

以英德翻译数据为例，模型训练完成后可以执行以下命令对指定文件中的文本进行翻译：

``` sh
# setting visible devices for prediction
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.1
# 执行 decoding_gemm 目的是基于当前环境、配置，提前确定一个性能最佳的矩阵乘算法，不是必要的步骤
cp -rf ../../../../paddlenlp/ops/build/third-party/build/fastertransformer/bin/decoding_gemm ./
./decoding_gemm 8 4 8 64 38512 32 512 0
python encoder_decoding_predict.py --config ../configs/transformer.base.yaml --decoding_lib ../../../../paddlenlp/ops/build/lib/libdecoding_op.so --decoding_strategy beam_search --beam_size 5
```

其中:
* `--config`: 选项用于指明配置文件的位置
* `--decoding_lib`: 选项用于指明编译好的 FasterTransformer decoding lib 的位置
* `--decoding_strategy`: 选项用于指定解码使用的策略，可以选择是 `beam_search`，`topk_sampling`，`topp_sampling`。
  * 当使用 `beam_search` 的时候，需要指定 `--beam_size` 的值
  * 当使用 `topk_sampling` 的时候，需要指定 `--topk` 的值
  * 当使用 `topp_sampling` 的时候，需要指定 `topp` 的值，并且需要保证 `--topk` 的值为 0
* `--beam_size`: 解码策略是 `beam_search` 的时候，beam size 的大小，数据类型是 `int`
* `--diversity_rate`: 解码策略是 `beam_search` 的时候，设置 diversity rate 的大小，数据类型是 `float`。当设置的 `diversity_rate` 大于 0 的时候，FasterTransformer 仅支持 beam size 为 1，4，16，64
* `--topk`: 解码策略是 `topk_sampling` 的时候，topk 计算的 k 值的大小，数据类型是 `int`
* `--topp`: 解码策略是 `topp_sampling` 的时候，p 的大小，数据类型是 `float`

翻译结果会输出到 `output_file` 指定的文件。执行预测时需要设置 `init_from_params` 来给出模型所在目录，更多参数的使用可以在 `./sample/config/transformer.base.yaml` 文件中查阅注释说明并进行更改设置。如果执行不提供 `--config` 选项，程序将默认使用 base model 的配置。


#### 使用动态图预测(使用 float16 decoding 预测)

float16 与 float32 预测的基本流程相同，不过在使用 float16 的 decoding 进行预测的时候，需要再加上 `--use_fp16_decoding` 选项，表示使用 fp16 进行预测。后按照与之前相同的方式执行即可。具体执行方式如下：

``` sh
# setting visible devices for prediction
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.1
# 执行 decoding_gemm 目的是基于当前环境、配置，提前确定一个性能最佳的矩阵乘算法，不是必要的步骤
cp -rf ../../../../paddlenlp/ops/build/third-party/build/fastertransformer/bin/decoding_gemm ./
./decoding_gemm 8 4 8 64 38512 32 512 1
python encoder_decoding_predict.py --config ../configs/transformer.base.yaml --decoding_lib ../../../../paddlenlp/ops/build/lib/libdecoding_op.so --use_fp16_decoding --decoding_strategy beam_search --beam_size 5
```

其中，`--config` 选项用于指明配置文件的位置，而 `--decoding_lib` 选项用于指明编译好的 FasterTransformer decoding lib 的位置。

翻译结果会输出到 `output_file` 指定的文件。执行预测时需要设置 `init_from_params` 来给出模型所在目录，更多参数的使用可以在 `./sample/config/transformer.base.yaml` 文件中查阅注释说明并进行更改设置。如果执行不提供 `--config` 选项，程序将默认使用 base model 的配置。

需要注意的是，目前预测仅实现了单卡的预测，原因在于，翻译后面需要的模型评估依赖于预测结果写入文件顺序，多卡情况下，目前暂未支持将结果按照指定顺序写入文件。

#### 导出基于 FasterTransformer 的预测库使用模型文件

我们提供一个已经基于动态图训练好的 base model 的 checkpoint 以供使用，当前 checkpoint 是基于 WMT 英德翻译的任务训练。可以通过[transformer-base-wmt_ende_bpe](https://bj.bcebos.com/paddlenlp/models/transformers/transformer/transformer-base-wmt_ende_bpe.tar.gz)下载。

使用 C++ 预测库，首先，我们需要做的是将动态图的 checkpoint 导出成预测库能使用的模型文件和参数文件。可以执行 `export_model.py` 实现这个过程。

``` sh
python export_model.py --config ../configs/transformer.base.yaml  --decoding_strategy beam_search --beam_size 5
```

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考 [文本生成高性能加速](../../../../paddlenlp/ops/README.md)。编译好后，可以在执行 `export_model.py` 时使用 `--decoding_lib ../../../../paddlenlp/ops/build/lib/libdecoding_op.so` 可以完成导入。

注意：如果是自行编译的话，这里的 `libdecoding_op.so` 的动态库是参照文档 [文本生成高性能加速](../../../../paddlenlp/ops/README.md) 中 **`Python 动态图使用自定义 op`** 编译出来的 lib，与相同文档中 **`C++ 预测库使用自定义 op`** 编译产出不同。因此，在使用预测库前，还需要额外导出模型：
  * 一次用于获取 Python 动态图下的 lib，用到 Python 端进行模型导出。
  * 一次获取编译的基于预测库的可执行文件

执行 `export_model.py` 之后，可以在当前路径的 `infer_model/` 下面看到导出的模型文件：
  ```text
  └── infer_model/
    ├── transformer.pdiparams
    └── transformer.pdmodel
  ```

#### C++ 预测库使用高性能加速

C++ 预测库使用 FasterTransformer 的高性能加速需要自行编译，可以参考 [文本生成高性能加速](../../../../paddlenlp/ops/README.md) 文档完成基于 C++ 预测库的编译，同时也可以参考相同文档执行对应的 C++ 预测库的 demo 完成预测。

具体的使用 demo 可以参考 [Transformer 预测库 C++ demo](../../../../paddlenlp/ops/faster_transformer/src/demo/transformer_e2e.cc)。

## 模型评估

评估方式与动态图评估方式相同，预测结果中每行输出是对应行输入的得分最高的翻译，对于使用 BPE 的数据，预测出的翻译结果也将是 BPE 表示的数据，要还原成原始的数据（这里指 tokenize 后的数据）才能进行正确的评估。评估过程具体如下（BLEU 是翻译任务常用的自动评估方法指标）：

``` sh
# 还原 predict.txt 中的预测结果为 tokenize 后的数据
sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt
# 若无 BLEU 评估工具，需先进行下载
git clone https://github.com/moses-smt/mosesdecoder.git
# 以英德翻译 newstest2014 测试数据为例
perl mosesdecoder/scripts/generic/multi-bleu.perl ~/.paddlenlp/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data/newstest2014.tok.de < predict.tok.txt
```

执行上述操作之后，可以看到类似如下的结果，此处结果是 beam_size 为 5 时 base model 在 newstest2014 上的 BLEU 结果：
```
BLEU = 26.89, 58.4/32.6/20.5/13.4 (BP=1.000, ratio=1.010, hyp_len=65166, ref_len=64506)
```

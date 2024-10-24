# 使用 Paddle Inference C++ API 推理

## 模型推理

通过前文介绍，我们可以获取导出后的预测模型。模型导出后，`infer_model/` 下的目录结构如下：

``` text
.
├── transformer.pdiparams
├── transformer.pdiparams.info
└── transformer.pdmodel
```

可以将存有导出后模型的目录拷贝到当前路径下：

``` sh
cp -rf ../../infer_model/ ./
```

使用 C++ 进行推理需要提前先编译出可执行文件。编译的方式可以直接使用 `run.sh`，不过需要做一些指定。

首先打开 run.sh：

``` sh
LIB_DIR=YOUR_LIB_DIR
CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
MODEL_DIR=YOUR_MODEL_DIR
VOCAB_DIR=/root/.paddlenlp/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708
DATA_DIR=/root/.paddlenlp/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en
```

需要依次指定：
* `LIB_DIR`: 所使用的 Paddle Inference 的库，即 `libpaddle_inference.so` 的位置。预测库的组织结构满足：
  ```text
  .
  ├── CMakeCache.txt
  ├── paddle/
    ├── include/
    └── lib/
  ├── third_party/
    ├── cudaerror/
    ├── install/
    └── threadpool/
  └── version.txt
  ```
* `CUDA_LIB_DIR`: 所使用的 CUDA 的库的位置。
* `CUDNN_LIB_DIR`: 所使用的 CUDNN 的库的位置。
* `MODEL_DIR`: 导出的模型的路径。
* `VOCAB_DIR`: 词表的位置。
* `DATA_DIR`: 需要推理的数据的位置，当前数据是经过 tokenize 以及 bpe 处理之后的序列用空格连接成的句子，并非原始数据。

可以简单执行如下语句完成编译以及推理整个过程。

``` sh
bash run.sh
```

以上步骤，如果全部正确执行，将会依次完成编译、预测全部过程。不过，如果需要自行执行可执行文件，编译完成后，其实，在 `build/bin/` 路径下会生成 `transformer_e2e` 的可执行文件，也可以直接执行这个可执行文件进行推理。

执行的参数及解释如下：

``` sh
export CUDA_VISIBLE_DEVICES=0
./build/bin/transformer_e2e -batch_size 8 -device gpu -gpu_id 0 -model_dir ./infer_model/ -vocab_file /root/.paddlenlp/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 -data_file /root/.paddlenlp/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en
```

各个参数解释如下：
* `-batch_size`: 使用 Paddle Inference 的时候一个 batch 的句子数目。
* `-device`: 使用的设备，可以是 gpu 或是 cpu。
* `-gpu_id`: 若使用 gpu，则需要提供所使用的 gpu 的 id。
* `-use_mkl`: 是否使用 mkl，设置代表使用 mkl，不设置则不使用 mkl。仅在使用 cpu 进行预测的时候有效。
* `-threads`: 仅在使用 mkl 的时候起效，用于指定计算 math 库时的线程数。
* `-model_dir`: 导出的模型的位置。
* `-vocab_file`: 词表文件的位置。
* `-data_file`: 推理用的数据的位置。

英德翻译的结果会保存到 `predict.txt` 文件中。

## 模型评估

推理结果中每行输出是对应行输入的得分最高的翻译，对于使用 BPE 的数据，预测出的翻译结果也将是 BPE 表示的数据，要还原成原始的数据（这里指 tokenize 后的数据）才能进行正确的评估。评估过程具体如下（BLEU 是翻译任务常用的自动评估方法指标）：

``` sh
# 还原 predict.txt 中的预测结果为 tokenize 后的数据
sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt
# 若无 BLEU 评估工具，需先进行下载
git clone https://github.com/moses-smt/mosesdecoder.git
# 以英德翻译 newstest2014 测试数据为例
perl mosesdecoder/scripts/generic/multi-bleu.perl ~/.paddlenlp/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data/newstest2014.tok.de < predict.tok.txt
```

执行上述操作之后，可以看到类似如下的结果，此处结果是 big model 在 newstest2014 上的 BLEU 结果：
```
BLEU = 27.48, 58.6/33.2/21.1/13.9 (BP=1.000, ratio=1.012, hyp_len=65312, ref_len=64506)
```

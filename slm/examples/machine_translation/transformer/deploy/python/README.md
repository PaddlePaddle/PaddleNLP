# 使用 Paddle Inference Python API 推理

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

执行如下命令可以使用 Paddle Inference Python API 进行推理：

``` sh
export CUDA_VISIBLE_DEVICES=0
python inference.py \
        --config ../../configs/transformer.base.yaml \
        --batch_size 8 \
        --device gpu \
        --model_dir ./infer_model/
```

各个参数解释如下：
* `--config`: yaml 配置文件，和训练时使用的相同，不过因为模型导出时已经固定了模型结构，因此，模型超参相关配置将不会再起作用，仅有 `reader` 相关配置、`infer_batch_size` 以及 `inference_model_dir` 仍会有效。
* `--batch_size`: 与配置文件中 `infer_batch_size` 意义相同，是指的使用 Paddle Inference 的时候一个 batch 的句子数目。
* `--device`: 使用的设备，可以是 gpu，xpu 或是 cpu。
* `--use_mkl`: 是否使用 mkl，没有设定表示不使用 mkl。可以通过 `--use_mkl True` 指定。
* `--threads`: 仅在使用 mkl 的时候起效，用于指定计算 math 库时的线程数。
* `--model_dir`: 导出的 Paddle Inference 可用的模型路径，与配置文件中的 `inference_model_dir` 对应。

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

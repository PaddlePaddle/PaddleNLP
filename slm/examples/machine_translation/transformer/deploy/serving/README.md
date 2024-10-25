# 使用 Paddle Serving 推理

## Paddle Serving 的使用

Paddle Serving 的安装可以参考[Paddle Serving 安装文档](https://github.com/PaddlePaddle/Serving#installation)。需要在服务端和客户端安装相关的依赖。

Serving 的执行包含两部分，其一是服务端的执行，其二是客户端的执行。接下来我们会一一说明如何使用 Paddle Serving 完成对 Transformer 的推理。

## 模型推理

通过前文介绍，我们可以获取导出后的预测模型。模型导出后，`infer_model/` 下的目录结构如下：

``` text
.
└── infer_model/
    ├── transformer.pdiparams
    ├── transformer.pdiparams.info
    └── transformer.pdmodel
```

可以将存有导出后模型的目录拷贝到当前路径下：

``` sh
cp -rf ../../infer_model/ ./
```

### 导出 Serving 模型和配置

使用导出的 Paddle Inference 的模型，我们需要再做一次转换，将上面保存在 `infer_model/` 下面的模型重新转换成 Paddle Serving 使用的模型。具体操作方式如下：

``` sh
python export_serving_model.py --model_dir ./infer_model/
```

执行结束之后，会在 shell 上打印出 Transformer 模型输入、输出的变量的名称：

``` sh
model feed_names : dict_keys(['src_word'])                          # 模型输入的变量的名称
model fetch_names : dict_keys(['save_infer_model/scale_0.tmp_1'])   # 模型输出的变量的名称
```

导出后，可以在当前路径下得到两个新的目录 `transformer_client/` 和 `transformer_server/`。

``` text
.
├── transformer_client/
    ├── serving_client_conf.prototxt
    └── serving_client_conf.stream.prototxt
└── transformer_server/
    ├── __model__
    ├── __params__
    ├── serving_server_conf.prototxt
    └── serving_server_conf.stream.prototxt
```

脚本成功执行并打印出预期内的输入、输出的变量的名称即完成整个过程。

### 启动服务端

Transformer 的服务端使用的是 Paddle Serving 的 `WebService` 相关接口。执行的命令如下：

``` sh
export CUDA_VISIBLE_DEVICES=0
python transformer_web_server.py --config ../../configs/transformer.base.yaml --device gpu --model_dir ./transformer_server
```

各个参数的解释如下：
* `--config`: yaml 配置文件，和训练时使用的相同，不过因为模型导出时已经固定了模型结构，因此，模型超参相关配置将不会再起作用，仅有 `reader` 相关配置，比如词表以及 `inference_model_dir` 等仍会有效。
* `--device`: 使用的设备，可以是 gpu 或是 cpu。
* `--model_dir`: 导出的 Paddle Serving 可用的模型路径，与配置文件中的 `inference_model_dir` 对应。在这里，特指的 `transformer_server/` 的路径。

### 启动客户端完成推理

在英德翻译的例子里面，在客户端这侧，我们只需要传给服务端需要翻译的句子即可。这里的句子是经过了 tokenize 以及 bpe 切词的序列用空格连接而成的句子。

执行的方式如下：

``` sh
python transformer_web_client.py --config ../../configs/transformer.base.yaml --batch_size 8
```

各个参数的解释如下：
* `--config`: yaml 配置文件，和训练时使用的相同，不过因为模型导出时已经固定了模型结构，因此，模型超参相关配置将不会再起作用，仅有 `reader` 相关配置，比如使用的测试集以及 `infer_batch_size` 等仍会有效。
* `--batch_size`: 与配置文件中 `infer_batch_size` 意义相同，是指的使用 Paddle Serving 的时候一个 batch 的句子数目。

执行完客户端的脚本，将会在本地生成一个 `predict.txt` 的文件，存有推理的结果。

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

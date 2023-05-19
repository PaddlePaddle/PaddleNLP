# 机器翻译

机器翻译（Machine Translation）是利用计算机将一种自然语言（源语言）转换为另一种自然语言（目标语言）的过程，输入为源语言句子，输出为相应的目标语言的句子。

## 快速开始

### 环境依赖

使用当前机器翻译示例，需要额外安装配置以下环境：

* attrdict
* pyyaml
* subword_nmt
* fastBPE (可选，若不使用 preprocessor.py 的 bpe 分词功能可以不需要)

### 数据准备

数据准备部分分成两种模式，一种是使用 PaddleNLP 内置的已经处理好的 WMT14 EN-DE 翻译的数据集，另一种，提供了当前 Transformer demo 使用自定义数据集的方式。以下分别展开介绍。

#### 使用内置已经处理完成数据集

内置的处理好的数据集是基于公开的数据集：WMT 数据集。

WMT 翻译大赛是机器翻译领域最具权威的国际评测大赛，其中英德翻译任务提供了一个中等规模的数据集，这个数据集是较多论文中使用的数据集，也是 Transformer 论文中用到的一个数据集。我们也将 [WMT'14 EN-DE 数据集](http://www.statmt.org/wmt14/translation-task.html) 作为示例提供。

可以编写如下代码，即可自动载入处理好的上述的数据，对应的 WMT14 EN-DE 的数据集将会自动下载并且解压到 `~/.paddlenlp/datasets/WMT14ende/`。

``` python
datasets = load_dataset('wmt14ende', splits=('train', 'dev'))
```

如果使用内置的处理好的数据，那到这里即可完成数据准备一步，可以直接移步 [Transformer 翻译模型](transformer/README.md) 将详细介绍如何使用内置的数据集训一个英德翻译的 Transformer 模型。

#### 使用自定义翻译数据集

本示例同时提供了自定义数据集的方法。可参考以下执行数据处理方式：

``` bash
# 数据下载、处理，包括 bpe 的训练
bash preprocessor/prepare-wmt14en2de.sh --icml17

# 数据预处理
DATA_DIR=examples/translation/wmt14_en_de

python preprocessor/preprocessor.py \
    --source_lang en \
    --target_lang de \
    --train_pref $DATA_DIR/train \
    --dev_pref $DATA_DIR/dev \
    --test_pref $DATA_DIR/test \
    --dest_dir data/wmt17_en_de \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --joined_dictionary
```

`preprocessor/preprocessor.py` 支持了在机器翻译中常见的数据预处理方式。在预处理 `preprocessor/preprocessor.py` 脚本中，则提供词表构建，数据集文件整理，甚至于 bpe 分词的功能（bpe 分词过程可选）。最后获取的处理完成的 train，dev，test 数据可以直接用于后面 Transformer 模型的训练、评估和推理中。具体的参数说明如下：

* `--src_lang`(`-s`): 指明数据处理对应的源语言类型，比如 `de` 表示德语，`en` 表示英语，`fr` 表示法语等等。
* `--trg_lang`(`-t`): 指明数据处理对应的目标语言的类型，比如 `de` 表示德语，`en` 表示英语，`fr` 表示法语等等。
* `--train_pref`: 指明前序步骤中，下载的训练数据的路径，以及对应的文件名前缀，比如 `preprocessor/wmt14_en_de/train` 结合 `--src_lang de` 和 `--trg_lang en`，表示在 `preprocessor/wmt14_en_de/` 路径下，源语言是 `preprocessor/wmt14_en_de/train.en`，目标语言是 `preprocessor/wmt14_en_de/train.de`。
* `--dev_pref`: 指明前序步骤中，下载的验证数据的路径，以及对应的文件名前缀。在验证集语料中，如果有的 token 在训练集中从未出现过，那么将会被 `<unk>` 替换。
* `--test_pref`: 指明前序步骤中，下载的测试数据的路径，以及对应的文件名前缀。在测试集语料中，如果有的 token 在训练集中从未出现过，那么将会被 `<unk>` 替换。
* `--dest_dir`: 完成数据处理之后，保存处理完成数据以及词表的路径。
* `--threshold_src`: 在源语言中，出现频次小于 `--threshold_src` 指定的频次的 token 将会被替换成 `<unk>`。默认为 0，表示不会根据 token 出现的频次忽略 token 本身。
* `--threshold_trg`: 在目标语言中，出现频次小于 `--threshold_trg` 指定的频次的 token 将会被替换成 `<unk>`。默认为 0，表示不会根据 token 出现的频次忽略 token 本身。
* `--src_vocab`: 源语言词表，默认为 None，表示需要预处理步骤根据训练集语料重新生成一份词表。如果源语言与目标语言共用同一份词表，那么将使用 `--src_vocab` 指定的词表。
* `--trg_vocab`: 目标语言词表，默认为 None，表示需要预处理步骤根据训练集语料重新生成一份词表。如果源语言与目标语言共用同一份词表，那么将使用 `--src_vocab` 指定的词表。
* `--nwords_src`: 源语言词表最大的大小，不包括 special token。默认为 None，表示不限制。若源语言和目标语言共用同一份词表，那么将使用 `--nwords_src` 指定的大小。
* `--nwords_trg`: 目标语言词表最大的大小，不包括 special token。默认为 None，表示不限制。若源语言和目标语言共用同一份词表，那么将使用 `--nwords_src` 指定的大小。
* `--align_file`: 是否将平行语料文件整合成一个文件。
* `--joined_dictionary`: 源语言和目标语言是否使用同一份词表。若不共用同一份词表，无需指定。
* `--only_source`: 是否仅处理源语言。
* `--dict_only`: 是否仅处理词表。若指定，则仅完成词表处理。
* `--bos_token`: 指明翻译所用的 `bos_token`，表示一个句子开始。
* `--eos_token`: 指明翻译所用的 `eos_token`，表示一个句子的结束。
* `--pad_token`: 指明 `pad_token`，用于将一个 batch 内不同长度的句子 pad 到合适长度。
* `--unk_token`: 指明 `unk_token`，用于当一个 token 在词表中未曾出现的情况，将使用 `--unk_token` 指明的字符替换。
* `--apply_bpe`: 是否需要对数据作 bpe 分词。若指定则会在 preprocessor.py 脚本开始执行 bpe 分词。如果是使用提供的 shell 脚本完成的数据下载，则无需设置，在 shell 脚本中会作 bpe 分词处理。
* `--bpe_code`: 若指明 `--apply_bpe` 使用 bpe 分词，则需同时提供训练好的 bpe code 文件。

除了 WMT14 德英翻译数据集外，我们也提供了其他的 shell 脚本完成数据下载处理，比如 WMT14 英法翻译数据。

``` bash
# WMT14 英法翻译的数据下载、处理
bash prepare-wmt14en2fr.sh
```

完成数据处理之后，同样也可以采用上文提到的预处理方式获取词表，完成预处理。

如果有或者需要使用其他的平行语料，可以自行完成下载和简单的处理。

在下载部分，即在 shell 脚本中，处理需要用到 [mosesdecoder](https://github.com/moses-smt/mosesdecoder) 和 [subword-nmt](https://github.com/rsennrich/subword-nmt) 这两个工具。包括:

* 使用 `mosesdecoder/scripts/tokenizer/tokenizer.perl` 完成对词做一个初步的切分；
* 基于 `mosesdecoder/scripts/training/clean-corpus-n.perl` 完成数据的清洗；
* 使用 `subword-nmt/subword_nmt/learn_bpe.py` 完成 bpe 的学习；

此外，基于学到的 bpe code 进行分词的操作目前提供了两种选项，其一是，可以在以上的 shell 脚本中处理完成，使用以下的工具：

* 使用 `subword-nmt/subword_nmt/apply_bpe.py` 完成分词工作。

其二，也可以直接在后面的 `preprocessor/preprocessor.py` 脚本中，指明 `--apply_bpe` 完成分词操作。


### 如何训一个翻译模型

前文介绍了如何快速开始完成翻译训练所需平行语料的准备，关于进一步的，模型训练、评估和推理部分，可以根据需要，参考对应的模型的文档：

* [Transformer 翻译模型](transformer/README.md)

## Acknowledge

我们借鉴了 facebookresearch 的 [fairseq](https://github.com/facebookresearch/fairseq) 在翻译数据的预处理上优秀的设计，在此对 fairseq 作者以及其开源社区表示感谢。

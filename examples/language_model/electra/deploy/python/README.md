# **ELECTRA 使用Paddle Inference API进行推理**

## 前提条件
准备好Inference所需模型，需要2个文件：
| 文件                          | 说明                                   |
|-------------------------------|----------------------------------------|
| electra-deploy.pdiparams      | 模型权重文件，供推理时加载使用            |
| electra-deploy.pdmodel        | 模型结构文件，供推理时加载使用            |

如何获得Inference模型？[可参考文档“导出推理模型”一节](../../README.md)，下面假设这2个文件已生成，并放在在当前目录下，有两种方法进行推理

## 从命令行读取输入数据进行推理
```shell
python -u ./predict.py \
    --model_file ./electra-deploy.pdmodel \
    --params_file ./electra-deploy.pdiparams \
    --predict_sentences "uneasy mishmash of styles and genres ." "director rob marshall went out gunning to make a great one ." \
    --batch_size 2 \
    --max_seq_length 128 \
    --model_name electra-small
```
其中参数释义如下：
- `model_file` 表示推理需要加载的模型结构文件。例如前提中得到的electra-deploy.pdmodel。
- `params_file` 表示推理需要加载的模型权重文件。例如前提中得到的electra-deploy.pdiparams。
- `predict_sentences` 表示用于推理的（句子）数据，可以配置1条或多条。如果此项配置，则predict_file不用配置。
- `batch_size` 表示每次推理的样本数目。
- `max_seq_length` 表示输入的最大句子长度，超过该长度将被截断。
- `model_name` 表示推理模型的类型，当前支持electra-small（约1400万参数）、electra-base（约1.1亿参数）、electra-large（约3.35亿参数）。

另外还有一些额外参数不在如上命令中：
- `use_gpu` 表示是否使用GPU进行推理，默认不开启。如果在命令中加上了--use_gpu，则使用GPU进行推理。
- `use_trt` 表示是否使用TensorRT进行推理，默认不开启。如果在命令中加上了--use_trt，且配置了--use_gpu，则使用TensorRT进行推理。前提条件：1）需提前安装TensorRT或使用[Paddle提供的TensorRT docker镜像](https://github.com/PaddlePaddle/Serving/blob/v0.5.0/doc/DOCKER_IMAGES_CN.md)。2）需根据cuda、cudnn、tensorRT和python的版本，安装[匹配版本的Paddle包](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html)

## 从文件读取输入数据进行推理
```shell
python -u ./predict.py \
    --model_file ./electra-deploy.pdmodel \
    --params_file ./electra-deploy.pdiparams \
    --predict_file "./sst-2.test.tsv.1" "./sst-2.test.tsv.2" \
    --batch_size 2 \
    --max_seq_length 128 \
    --model_name electra-small
```
其中绝大部分和从命令行读取输入数据一样，这里描述不一样的参数：
- `predict_file` 表示用于推理的文件数据，可以配置1个或多个文件，每个文件和预训练数据格式一样，为utf-8编码的文本数据，每行1句文本。如果此项配置，则predict_sentences不用配置。

模型对每1句话分别推理出1个结果，例如下面为使用第一种方法中的命令得到的SST-2情感分类推理结果，0表示句子是负向情感，1表示句子为正向情感。因为batch_size=2，所以只有1个batch：
```shell
===== batch 0 =====
Input sentence is : [CLS] uneasy mishmash of styles and genres . [SEP]
Output data is : 0
Input sentence is : [CLS] director rob marshall went out gunning to make a great one . [SEP]
Output data is : 1
inference total 2 sentences done, total time : 0.0849156379699707 s
```
此推理结果表示：第1句话是负向情感，第2句话是正向情感。

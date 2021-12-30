# LayoutXLM

## 模型简介
本项目是 [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/pdf/2104.08836.pdf) 在 Paddle 2.2上的开源实现，
包含了在 [XFUND数据集](https://github.com/doc-analysis/XFUND) 上的微调代码。

## 快速开始
### 配置环境
环境依赖
- cv2
- sentencepiece
- yacs

安装命令：
```shell
pip install opencv-python
pip install sentencepiece
pip install yacs
```

### 数据准备
处理好的XFUND中文数据集下载地址：https://bj.bcebos.com/v1/paddlenlp/datasets/XFUND.zip 。

下载并解压该数据集，解压后将数据集放置在当前目录下。

### 执行Fine-tuning
1. ``Semantic Entity Recognition`` 任务启动Fine-tuning的方式如下：
    ```shell
    bash run_xfun_ser.sh

    # 结果如下:
    # best metrics: {'precision': 0.8514686248331108, 'recall': 0.9354602126879354, 'f1': 0.8914904770225406}
    ```

2. ``Relation Extraction`` 任务启动Fine-tuning的方式如下：
    ```shell
    bash run_xfun_re.sh

    # 结果如下:
    # best metrics: {'precision': 0.6788935658448587, 'recall': 0.7743484224965707, 'f1': 0.7234860621595642}
    ```

## Reference
- [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/pdf/2104.08836.pdf)
- [microsoft/unilm/layoutxlm](https://github.com/microsoft/unilm/tree/master/layoutxlm)

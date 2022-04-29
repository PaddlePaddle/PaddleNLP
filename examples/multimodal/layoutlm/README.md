# LayoutLM

## 模型简介
本项目是 [LayoutLM:Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/pdf/1912.13318v5.pdf) 在 Paddle 2.2上的开源实现，
包含了在 [FUNSD数据集](https://guillaumejaume.github.io/FUNSD/) 上的微调代码。

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
处理好的FUNSD中文数据集下载地址：https://bj.bcebos.com/v1/paddlenlp/datasets/FUNSD.zip 。

下载并解压该数据集，解压后将数据集放置在当前目录下。

### 执行Fine-tuning
1. ``Sequence Labeling`` 任务启动Fine-tuning的方式如下：
    ```shell
    bash train_funsd.sh

    # 结果如下:
    # best metrics: {'precision': 0.7642124883504194, 'recall': 0.8204102051025512, 'f1': 0.7913148371531967}
    ```

### 数据处理
FUNSD数据集是常用的表格理解数据集，原始的数据集下载地址：https://guillaumejaume.github.io/FUNSD/dataset.zip.
包括training_data和test_dataing两个子文件夹，包括149个训练数据和50个测试数据。数据预处理方式如下：
```shell
    bash preprocess.sh
```

## Reference
- [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/pdf/1912.13318v5.pdf)
- [microsoft/unilm/layoutlm](https://github.com/microsoft/unilm/tree/master/layoutlm)

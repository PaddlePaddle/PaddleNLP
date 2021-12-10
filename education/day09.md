# Day09 机器同传作业辅导

本次AI Studio课程课后作业为实现机器同传的Demo，主要包括文本同传和语音同传（加分项）。  
下面给出实现的步骤，具体细节参考[教程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/simultaneous_translation/stacl/demo) 。

## 1. 克隆PaddleNLP代码库
```bash
# 克隆代码
git clone https://github.com/PaddlePaddle/PaddleNLP.git
# 进入demo目录
cd PaddleNLP/examples/simultaneous_translation/stacl/demo
```

## 2. 配置预训练模型
在demo目录下创建models目录，并在models目录下创建下面四个waitk策略的子目录：
- nist_wait_1
- nist_wait_3
- nist_wait_5
- nist_wait_-1  

在此[下载预训练模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/simultaneous_translation/stacl/README.md#%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD%E6%9B%B4%E6%96%B0%E4%B8%AD) ，下载完后将解压后的`transformer.pdparams`分别放在各个waitk策略对应的子目录下。

## 3. 配置词表
下载词表文件到demo目录下，词表文件为：
- [中文BPE词表下载](https://bj.bcebos.com/paddlenlp/models/stacl/2M.zh2en.dict4bpe.zh)
- [source vocab](https://bj.bcebos.com/paddlenlp/models/stacl/nist.20k.zh.vocab)
- [target vocab](https://bj.bcebos.com/paddlenlp/models/stacl/nist.10k.en.vocab)

## 4. 配置依赖环境

### 4.1 基本环境
```bash
# 通过以下命令安装
pip install -r requirements.txt
```
注意：本项目依赖于Python内置包`tkinter >= 8.6`

### 4.2 语音同传环境（本地麦克风收音）
需要安装`pyaudio==0.2.11`来调用本地麦克风，安装教程参考[官网](http://people.csail.mit.edu/hubert/pyaudio/) 。  
安装失败，则只会启动文本同传。

### 4.3 语音同传环境（百度AI语音识别）
需要配置`const.py`里面语音识别的应用鉴权信息，只需要将`APPID`和`APPKEY`设置为自己所申请的。  
申请教程：[教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/simultaneous_translation/stacl/demo/README_ai.md)

## 5. 运行
- 1.在Chinese input输入中文，按【回车键】开始实时翻译，遇到【。！？】结束整句，按【CLEAR】清空所有的输入和输出；
- 2.按【REC】开始录音并实时翻译，遇到【。！？】结束整句，按【CLEAR】清空所有的输入和输出。

## ⚠️注意事项
- 1.demo需在本地电脑运行（需要GUI界面），已在Mac/Windows系统上通过测试；
- 2.语音同传为加分项，故若语音同传环境安装失败则只启动文本同传；
- 3.文本同传和语音同传交替使用之前，务必按【CLEAR】清空上一状态。

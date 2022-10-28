# ERNIE-ViLG 文生图系统

## 1. 场景概述

ERNIE-ViLG是一个知识增强跨模态图文生成大模型，将文生成图和图生成文任务融合到同一个模型进行端到端的学习，从而实现文本和图像的跨模态语义对齐。可以支持用户进行内容创作，让每个用户都能够体验到一个低门槛的创作平台。更多详细信息请参考官网的介绍[ernieVilg](https://wenxin.baidu.com/moduleApi/ernieVilg)


## 2. 产品功能介绍

本项目提供了低成本搭建端到端文生图的能力。用户需要进行简单的参数配置，然后输入prompts就可以生成各种风格的画作，另外，Pipelines提供了 Web 化产品服务，让用户在本地端就能搭建起来文生图系统。

<div align="center">
    <img src="https://user-images.githubusercontent.com/12107462/198007539-51863b31-715c-4cf4-921a-9ddf036c036b.gif" width="500px">
</div>


## 3. 快速开始: 快速搭建文生图系统


### 3.1 运行环境和安装说明

本实验采用了以下的运行环境进行，详细说明如下，用户也可以在自己的环境进行：

a. 软件环境：
- python >= 3.7.0
- paddlenlp >= 2.4.0
- paddlepaddle-gpu >=2.3
- CUDA Version: 10.2
- NVIDIA Driver Version: 440.64.00
- Ubuntu 16.04.6 LTS (Docker)

b. 硬件环境：

- NVIDIA Tesla V100 16GB x4卡
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

c. 依赖安装：
首先需要安装PaddlePaddle，PaddlePaddle的安装请参考文档[官方安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)，然后安装下面的依赖：
```bash
# pip 一键安装
pip install --upgrade paddle-pipelines -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或者源码进行安装最新版本
cd ${HOME}/PaddleNLP/pipelines/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py install
```
【注意】以下的所有的流程都只需要在`pipelines`根目录下进行，不需要跳转目录；另外，文生图系统需要联网，用户需要在有网的环境下进行。


### 3.2 一键体验文生图系统

在运行下面的命令之前，需要在[ERNIE-ViLG官网](https://wenxin.baidu.com/moduleApi/ernieVilg)申请`API Key`和 `Secret key`两个密钥(需要登录，登录后点击右上角的查看AK/SK，具体如下图)，然后执行下面的命令。

<div align="center">
    <img src="https://user-images.githubusercontent.com/12107462/196942735-06953270-ce1e-45a5-9e0d-5841068a8464.png" width="500">
</div>


#### 3.2.1 快速一键启动

您可以通过如下命令快速体验文生图系统的效果
```bash
python examples/text_to_image/text_to_image_example.py --prompt_text 宁静的小镇 \
                                                       --style 古风 \
                                                       --topk 5 \
                                                       --api_key 你申请的apikey \
                                                       --secret_key 你申请的secretkey \
                                                       --output_dir ernievilg_output
```
大概运行一分钟后就可以得到结果了,生成的图片请查看您的输出目录`output_dir`。

### 3.3 构建 Web 可视化文生图系统

整个 Web 可视化文生图系统主要包含 2 大组件: 1. 基于 RestfulAPI 构建模型服务 2. 基于 Gradio 构建 WebUI，接下来我们依次搭建这 2 个服务并最终形成可视化的文生图系统。

#### 3.3.1 启动 RestAPI 模型服务

启动之前，需要把您申请的`API Key`和 `Secret key`两个密钥添加到`text_to_image.yaml`的ak和sk的位置，然后运行：

```bash
export PIPELINE_YAML_PATH=rest_api/pipeline/text_to_image.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```
Linux 用户推荐采用 Shell 脚本来启动服务：：

```bash
sh examples/text_to_image/run_text_to_image.sh
```

#### 3.3.2 启动 WebUI

WebUI使用了[gradio前端](https://gradio.app/)，首先需要安装gradio，运行命令如下：
```
pip install gradio
```
然后使用如下的命令启动：
```bash
# 配置模型服务地址
export API_ENDPOINT=http://127.0.0.1:8891
# 在指定端口 8502 启动 WebUI
python ui/webapp_text_to_image.py --serving_port 8502
```
Linux 用户推荐采用 Shell 脚本来启动服务：：

```bash
sh examples/text_to_image/run_text_to_image_web.sh
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验文生图系统服务了。

如果安装遇见问题可以查看[FAQ文档](../../FAQ.md)

## Acknowledge

我们借鉴了 Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) 优秀的框架设计，在此对[Haystack](https://github.com/deepset-ai/haystack)作者及其开源社区表示感谢。

We learn form the excellent framework design of Deepset.ai [Haystack](https://github.com/deepset-ai/haystack), and we would like to express our thanks to the authors of Haystack and their open source community.

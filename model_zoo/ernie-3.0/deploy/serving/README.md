# FastDeploy ERNIE 3.0 模型 Serving 部署示例


在服务化部署前，需确认

- 1. 服务化镜像的软硬件环境要求和镜像拉取命令请参考[FastDeploy服务化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/README_CN.md)

## 环境准备


## 准备模型

模型移动好之后，分类任务的 models 目录结构如下:

```
models
├── ernie_seqcls                      # 分类任务的pipeline
│   ├── 1
│   └── config.pbtxt                  # 通过这个文件组合前后处理和模型推理
├── ernie_seqcls_model                # 分类任务的模型推理
│   ├── 1
│   │   ├── model.pdiparams
│   │   └── model.pdmodel
│   └── config.pbtxt
├── ernie_seqcls_postprocess          # 分类任务后处理
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── ernie_tokenizer                   # 预处理分词
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

序列标注任务的 models 目录结构如下:

```
models
├── ernie_tokencls                      # 序列标注任务的 pipeline
│   ├── 1
│   └── config.pbtxt                    # 通过这个文件组合前后处理和模型推理
├── ernie_tokencls_model                # 序列标注任务的模型推理
│   ├── 1
│   │   ├── model.pdiparams
│   │   └── model.pdmodel
│   └── config.pbtxt
├── ernie_tokencls_postprocess          # 序列标注任务后处理
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── ernie_tokenizer                     # 预处理分词
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

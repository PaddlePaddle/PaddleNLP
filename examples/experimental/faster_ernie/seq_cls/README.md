# FasterErnie实现文本分类

为了更好地实现训推一体化，PaddleNLP 2.2版本将文本预处理FasterTokenizer内置到Ernie模型内形成FasterErnie模型。

以下示例展示FasterErnie用于文本分类任务。

## 训练

```shell
export CUDA_VISIBLE_DEVICES=0
python train.py --device gpu --save_dir ckpt/
```

## 预测

```shell
export CUDA_VISIBLE_DEVICES=0
python predict.py --device gpu --save_dir ckpt/model_900
```

## 导出模型

```shell
python export_model.py --save_path ckpt/model_900 --output_path export/ --max_seq_length 128
```

## 部署推理

### python端

```shell
export CUDA_VISIBLE_DEVICES=0
python python_deploy.py --model_dir export/ --batch_size 1
```

### cpp端

首先需要从[官网](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html)下载2.2 版本以上paddle inference lib。
解压至cpp_deploy/lib文件目录下, 如
```
cpp_deploy/lib
├── CMakeLists.txt
└── paddle_inference
    ├── CMakeCache.txt
    ├── paddle
    │   ├── include                                    C++ 预测库头文件目录
    │   │   ├── crypto
    │   │   ├── internal
    │   │   ├── paddle_analysis_config.h
    │   │   ├── paddle_api.h
    │   │   ├── paddle_infer_declare.h
    │   │   ├── paddle_inference_api.h                 C++ 预测库头文件
    │   │   ├── paddle_mkldnn_quantizer_config.h
    │   │   └── paddle_pass_builder.h
    │   └── lib
    │       ├── libpaddle_inference.a                  C++ 静态预测库文件
    │       └── libpaddle_inference.so                 C++ 动态态预测库文件
    ├── third_party
    │   ├── install                                    第三方链接库和头文件
    │   │   ├── cryptopp
    │   │   ├── gflags
    │   │   ├── glog
    │   │   ├── mkldnn
    │   │   ├── mklml
    │   │   ├── protobuf
    │   │   ├── utf8proc
    │   │   └── xxhash
    │   └── threadpool
    │       └── ThreadPool.h
    └── version.txt
```


运行方式：

```shell
sh run.sh
```

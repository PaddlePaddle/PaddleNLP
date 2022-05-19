# **ELECTRA 使用Paddle Lite API进行推理**
在移动设备（手机、平板等）上需要使用Paddle Lite进行推理。[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)是飞桨轻量化推理引擎，为手机、IOT端提供高效推理能力，并广泛整合跨平台硬件，为端侧部署及应用落地问题提供轻量化的部署方案。下面以Android手机(Armv7或Armv8)为例，使用Paddle Lite进行ELECTRA模型的推理。

## 前提条件
准备好Inference所需模型，需要2个文件：
| 文件                          | 说明                                   |
|-------------------------------|----------------------------------------|
| electra-deploy.pdiparams      | 模型权重文件，供推理时加载使用            |
| electra-deploy.pdmodel        | 模型结构文件，供推理时加载使用            |

如何获得Inference模型？[可参考文档“导出推理模型”一节](../../README.md)，下面假设这2个文件已生成，并放在在当前目录下

## 准备硬件和系统
- 电脑。用于保存代码和数据；编译Paddle Lite（看需要）
- 手机。Android手机(Armv7或Armv8)，手机要能直接连接电脑，或者手机直连某个设备，其能连接到电脑。

如果在其它特殊硬件上或想要自己编译Paddle Lite预测库和优化工具，则电脑上还需准备：
- 交叉编译环境。不同开发环境的编译流程请参考对应文档。
   - [Docker](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#docker)
   - [Linux](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#linux)
   - [MAC OS](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#mac-os)

## 准备Paddle Lite预测库
有两种方法：
- 直接下载。[官方预测库下载地址](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html)，注意选择和手机arm系统版本匹配的，并带with_extra=ON的下载链接。
- 编译Paddle-Lite得到预测库。**需要先准备好交叉编译环境**，然后依次执行如下命令，例如编译在 armv8 硬件上运行的预测库并打开extra op：
```shell
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git checkout develop
./lite/tools/build_android.sh --arch=armv8 --with_extra=ON
```
直接下载预测库并解压后，可以得到`inference_lite_lib.android.armv8/`文件夹，通过编译Paddle-Lite得到的预测库位于`Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/`文件夹。无论使用哪种方法，得到的预测库的文件目录结构都如下，为了方便统一说明，下面都假设预测库位于${Paddle_Lite_root}/inference_lite_lib.android.armv8/目录中：
```
${Paddle_Lite_root}/inference_lite_lib.android.armv8/
|-- cxx                                        C++ 预测库和头文件
|   |-- include                                C++ 头文件
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib                                           C++预测库
|       |-- libpaddle_api_light_bundled.a             C++静态库
|       `-- libpaddle_light_api_shared.so             C++动态库
|-- java                                     Java预测库
|   |-- jar
|   |   `-- PaddlePredictor.jar
|   |-- so
|   |   `-- libpaddle_lite_jni.so
|   `-- src
|-- demo                                     C++和Java示例代码
|   |-- cxx                                  C++  预测库demo
|   `-- java                                 Java 预测库demo
```

## 准备Paddle Lite模型优化工具
因为移动设备上对模型的要求很严格，所以需要使用Paddle Lite模型优化工具将Inference模型优化后才能将模型部署到移动设备上进行推理，模型优化的方法包括量化、子图融合、混合调度、Kernel优选等等方法。准备Paddle Lite模型优化工具也有两种方法：
- 直接下载。
```shell
pip install paddlelite。
```
- 编译Paddle-Lite得到模型优化工具。**需要先准备好交叉编译环境**，然后依次执行如下命令：
```shell
# 如果准备环境时已经clone了Paddle-Lite，则不用重新clone Paddle-Lite
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git checkout develop
# 启动编译
./lite/tools/build.sh build_optimize_tool
```
如果是直接下载，工具可执行文件为`paddle_lite_opt`，并放在系统环境变量PATH中，所以无需进入到工具所在目录就可以直接执行；如果是编译得到，则工具可执行文件为`Paddle-Lite/build.opt/lite/api/opt`，为了后面统一说明，可将工具统一命名为`paddle_lite_opt`，并将其所处目录添加到系统环境变量PATH中，通过如下方式查看其运行选项和使用方式；
```shell
cd build.opt/lite/api/ && mv opt paddle_lite_opt
./paddle_lite_opt
```

## 使用Paddle Lite模型优化工具转换Inference模型
以前提条件中准备好的Inference模型 electra-deploy.pdmodel/electra-deploy.pdiparams 为例，执行：
```shell
paddle_lite_opt \
    --model_file ./electra-deploy.pdmodel \
    --param_file ./electra-deploy.pdiparams \
    --optimize_out ./electra-deploy-lite \
    --optimize_out_type protobuf \
    --valid_targets arm \
    --record_tailoring_info false
```
其中参数释义如下：
- `model_file` 表示推理需要加载的模型结构文件。例如前提中得到的electra-deploy.pdmodel。
- `param_file` 表示推理需要加载的模型权重文件。例如前提中得到的electra-deploy.pdiparams。
- `optimize_out` 表示输出的Lite模型**名字前缀**。例如配置./electra-deploy-lite，最终得到的Lite模型为./electra-deploy-lite.nb。
- `optimize_out_type` 表示输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf。
- `valid_targets` 表示模型将要运行的硬件类型，默认为arm。目前可支持x86、arm、opencl、npu、xpu，可以同时指定多个backend(以空格分隔)，Model Optimize Tool将会自动选择最佳方式。如果需要支持华为NPU（Kirin 810/990 Soc搭载的达芬奇架构NPU），应当设置为npu, arm。
- `record_tailoring_info` 表示是否使用 根据模型裁剪库文件 功能，如使用则设置该选项为true，以记录优化后模型含有的kernel和OP信息，默认为false。

如上命令执行后，得到Lite模型为./electra-deploy-lite.nb

## 预处理输入数据，并和Lite预测库、Lite模型、编译好的C++代码/配置 一起打包。
```shell
# 假设${Paddle_Lite_root}已经配置了正确的Lite预测库路径
python -u ./prepare.py \
    --lite_lib_path ${Paddle_Lite_root}/inference_lite_lib.android.armv8/ \
    --lite_model_file ./electra-deploy-lite.nb \
    --predict_file ./test.txt \
    --max_seq_length 128 \
    --model_name electra-small

# 进入lite demo的工作目录
cd ${Paddle_Lite_root}/inference_lite_lib.android.armv8/demo/cxx/electra/
make -j && mv electra_lite debug
```
其中prepare.py的参数释义如下：
- `lite_lib_path` 表示预测库所在目录。
- `lite_model_file` 表示Lite模型路径。
- `predict_file` 表示用于推理的文件数据，可以配置1个或多个文件，每个文件和预训练数据格式一样，为utf-8编码的文本数据，每行1句文本。
- `max_seq_length` 表示输入的最大句子长度，超过该长度将被截断。
- `model_name` 表示推理模型的类型，当前支持electra-small（约1400万参数）、electra-base（约1.1亿参数）、electra-large（约3.35亿参数）。

如上命令执行完后，${Paddle_Lite_root}/inference_lite_lib.android.armv8/demo/cxx/electra/文件夹下将有如下文件，只有其中的**debug目录**会传到手机：
```shell
demo/cxx/electra/
|-- debug/
|    |--config.txt                       推理配置和超参数配置
|    |--electra-deploy-lite.nb           优化后的Lite模型文件
|    |--electra_lite                     编译好的在手机上执行推理的可执行文件
|    |--libpaddle_light_api_shared.so    C++预测库文件
|    |--predict_input.bin                预处理好的输入数据（二进制）
|    |--predict_input.txt                输入数据明文
|    |--sst2_label.txt                   类别说明文件
|-- config.txt                              推理配置和超参数配置
|-- Makefile                                编译文件
|-- sentiment_classfication.cpp                推理代码文件
```

## 与目标设备连接执行推理
如果电脑和Android手机直接连接，则在电脑上安装[ADB工具](https://developer.android.com/studio/command-line/adb)，通过ADB工具来连接和操作Android设备：
```shell
# 检查是否连接上设备
adb devices
# 将debug目录推送到设备的/data/local/tmp/electra/目录下，需事先在设备上创建
adb push debug /data/local/tmp/electra/
# 登录设备并打开设备上的shell
adb shell
# 准备相关环境。进入程序目录，配置好动态链接库的环境变量并给程序添加执行权限
cd /data/local/tmp/electra/debug && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/electra/debug/ && chmod +x electra_lite
# 输入数据，运行Lite推理
./electra_lite ./config.txt
```
如果电脑和Android手机没有直接连接，Android手机直连某个设备，则需将debug目录cp到那个设备上，并在那个设备上安装ADB工具以执行如上代码。

执行如上推理命令后得到如下结果，同样数据在Paddle Lite推理的结果应该和使用Inference/Serving的结果是一样的
```shell
=== electra predict result: ./predict_input.txt===
sentence: [CLS] uneasy mishmash of styles and genres . [SEP], class_id: 0(negative), logits: 2.22824
sentence: [CLS] director rob marshall went out gunning to make a great one . [SEP], class_id: 1(positive), logits: 0.241332
total time : 0.399562 s.
```

如果修改了代码，则需要先执行prepare.py，再重新编译并打包push到手机上；如果只修改输入数据，则只需要执行prepare.py并打包push到手机上，不用重新编译。

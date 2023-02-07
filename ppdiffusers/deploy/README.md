# FastDeploy Stable Diffusion 模型高性能部署

本示例展现如何通过 FastDeploy 将我们 PPDiffusers 训练好的 Stable Diffusion 模型进行多硬件多推理引擎后端高性能部署。

## 部署模型准备

本示例需要使用训练模型导出后的部署模型，可参考[模型导出文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/deploy/export.md)导出部署模型。

## 环境依赖

在示例中使用了 FastDeploy，需要执行以下命令安装依赖。

```shell
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

## 快速体验

我们经过部署模型准备，可以开始进行测试。本目录提供 StableDiffusion 模型支持的三种任务，分别是文图生成、文本引导的图像变换以及文本引导的图像编辑。

### 文图生成 （Text-to-Image Generation）


下面将指定模型目录以及推理引擎后端，运行`text_to_img_infer.py`脚本，完成文图生成任务。

```
python text_to_img_infer.py --model_dir stable-diffusion-v1-4/ --scheduler "pndm" --backend paddle
```

得到的图像文件为fd_astronaut_rides_horse.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![fd_astronaut_rides_horse.png](https://user-images.githubusercontent.com/10826371/200261112-68e53389-e0a0-42d1-8c3a-f35faa6627d7.png)

如果使用stable-diffusion-v1-5模型，则可执行以下命令完成推理：

```
python text_to_img_infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler_ancestral" --backend paddle
```

#### 参数说明

`text_to_img_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。 |
| --model_format | 模型格式。默认为`'paddle'`，可选列表：`['paddle', 'onnx']`。 |
| --backend | 推理引擎后端。默认为`paddle`，可选列表：`['onnx_runtime', 'paddle', 'paddlelite']`，当模型格式为`onnx`时，可选列表为`['onnx_runtime']`。 |
| --device | 运行设备。默认为`gpu`，可选列表：`['cpu', 'gpu', 'huawei_ascend_npu', 'kunlunxin_xpu']`。 |
| --scheduler | StableDiffusion 模型的scheduler。默认为`'pndm'`。可选列表：`['pndm', 'euler_ancestral']`，StableDiffusio模型对应的scheduler可参考[ppdiffuser模型列表](https://github.com/PaddlePaddle/PaddleNLP/tree/main/ppdiffusers#ppdiffusers%E6%A8%A1%E5%9E%8B%E6%94%AF%E6%8C%81%E7%9A%84%E6%9D%83%E9%87%8D)。|
| --unet_model_prefix | UNet模型前缀。默认为`unet`。 |
| --vae_model_prefix | VAE模型前缀。默认为`vae_decoder`。 |
| --text_encoder_model_prefix | TextEncoder模型前缀。默认为`text_encoder`。 |
| --inference_steps | UNet模型运行的次数，默认为100。 |
| --image_path | 生成图片的路径。默认为`fd_astronaut_rides_horse.png`。  |
| --device_id | gpu设备的id。若`device_id`为-1，视为使用cpu推理。 |
| --use_fp16 | 是否使用fp16精度。默认为`False`。使用tensorrt或者paddle-tensorrt后端时可以设为`True`开启。 |

### 文本引导的图像变换（Image-to-Image Text-Guided Generation）



### 文本引导的图像编辑（Text-Guided Image Inpainting）

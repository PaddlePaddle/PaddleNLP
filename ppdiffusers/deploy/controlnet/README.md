# FastDeploy ControlNet 模型高性能部署

 **目录**
   * [部署模型准备](#部署模型准备)
   * [环境依赖](#环境依赖)
   * [快速体验](#快速体验)
       * [Canny to Image](#Canny-to-Image)


⚡️[FastDeploy](https://github.com/PaddlePaddle/FastDeploy) 是一款全场景、易用灵活、极致高效的AI推理部署工具，为开发者提供多硬件、多推理引擎后端的部署能力。开发者只需调用一行代码即可随意切换硬件、推理引擎后端。本示例展现如何通过 FastDeploy 将我们 PPDiffusers 训练好的 Stable Diffusion 模型进行多硬件、多推理引擎后端高性能部署。

<a name="部署模型准备"></a>

## 部署模型准备

本示例需要使用训练模型导出后的部署模型，可参考[模型导出文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/deploy/controlnet/export.md)导出部署模型。

<a name="环境依赖"></a>

## 环境依赖

在示例中使用了 FastDeploy，需要执行以下命令安装依赖。

```shell
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

<a name="快速体验"></a>

## 快速体验

我们经过部署模型准备，可以开始进行测试。本目录提供采用Canny边缘检测图片作为控制条件生成图片的教程 。

<a name="Canny to Image"></a>

### Canny to Image
下面左图是我们提供的初始图片，右图是经过OpenCV中的Canny算法处理后得到的边缘检测图片。

![bird](https://user-images.githubusercontent.com/50394665/225192117-3ec7a61c-227b-4056-a076-d37759f8411b.png)
![control_bird_canny](https://user-images.githubusercontent.com/50394665/225192606-47ba975f-f6cc-4555-8d85-870dc1327b45.png)

> Tips：为了能够跑出最快的推理速度，如果是使用`A卡GPU`的用户，请保证`低于8.5版本的TRT`不在`LD_LIBRARY_PATH`路径上。

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `controlnet_infer.py` 脚本，完成 `Canny to Image` 任务。

```sh
python controlnet_infer.py --model_dir ./control_sd15_canny --scheduler "pndm" --backend paddle
```

脚本的输入提示语句为 **"bird"**， 得到的图像文件为 `controlnet_bird.png`。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![controlnet_bird](https://user-images.githubusercontent.com/50394665/224997460-81b82457-d9ee-485f-9bc7-d703a716a2de.png)

如果使用 `paddle_tensorrt` 推理引擎后端及开启`半精度推理`，则可执行以下命令完成推理：

```sh
python controlnet_infer.py --model_dir control_sd15_canny --scheduler "euler_ancestral" --backend paddle_tensorrt --device gpu --benchmark_steps 10 --use_fp16 True
```

经测试，使用上述命令，在 80G A100 机器上能够跑出 `1.111716 s` 的成绩。

#### 参数说明

`controlnet_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。

<details><summary>&emsp; 命令行参数说明 </summary>

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。 |
| --model_format | 模型格式。默认为 `'paddle'`，可选列表：`['paddle', 'onnx']`。 |
| --backend | 推理引擎后端。默认为 `paddle`，可选列表：`['onnx_runtime', 'paddle', 'paddlelite', 'paddle_tensorrt']`，当模型格式为 `onnx` 时，可选列表为 `['onnx_runtime']`。 |
| --device | 运行设备。默认为 `cpu`，可选列表：`['cpu', 'gpu', 'huawei_ascend_npu', 'kunlunxin_xpu']`。 |
| --scheduler | ControlNet 模型的 scheduler。默认为 `'euler_ancestral'`。可选列表：`['pndm', 'euler_ancestral']`。|
| --unet_model_prefix | UNet 模型前缀。默认为 `unet`。 |
| --vae_decoder_model_prefix | VAE Decoder 模型前缀。默认为 `vae_decoder`。 |
| --vae_encoder_model_prefix | VAE Encoder 模型前缀。默认为 `vae_encoder`。 |
| --text_encoder_model_prefix | Text Encoder 模型前缀。默认为 `text_encoder`。 |
| --low_threshold | Canny算法最后一步中，小于该阈值的像素直接置为0，默认为 100。 |
| --high_threshold | Canny算法最后一步中，大于该阈值的像素直接置为255，默认为 200。 |
| --resolution | 想要生成的图片分辨率，默认为 512。 |
| --inference_steps | UNet 模型运行的次数，默认为 50。 |
| --image_path | 生成图片的路径。默认为 `controlnet_bird.png`。  |
| --device_id | gpu 设备的 id。若 `device_id` 为-1，视为使用 cpu 推理。 |
| --use_fp16 | 是否使用 fp16 精度。默认为 `False`。使用 tensorrt 或者 paddle_tensorrt 后端时可以设为 `True` 开启。 |

</details>

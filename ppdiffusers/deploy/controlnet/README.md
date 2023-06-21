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
python controlnet_infer.py --model_dir ./control_sd15_canny --scheduler "ddim" --backend paddle --task text2img_control
```

脚本的输入提示语句为 **"bird"**， 得到的图像文件为 `text2img_control.png`。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![text2img_control](https://user-images.githubusercontent.com/50394665/224997460-81b82457-d9ee-485f-9bc7-d703a716a2de.png)

如果使用 `paddle_tensorrt` 推理引擎后端及开启`半精度推理`，则可执行以下命令完成推理：

```sh
python controlnet_infer.py --model_dir control_sd15_canny --scheduler "preconfig-euler-ancestral" --backend paddle_tensorrt --device gpu --benchmark_steps 10 --use_fp16 True --task text2img_control
```

经测试，使用上述命令，在 80G A100 机器上能够跑出 `1.111716 s` 的成绩。

#### 参数说明

`controlnet_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。默认为 `runwayml/stable-diffusion-v1-5-canny@fastdeploy` |
| --backend | 推理引擎后端。默认为 `paddle_tensorrt`，可选列表：`['onnx_runtime', 'paddle', 'paddlelite', 'paddle_tensorrt', 'tensorrt']`。 |
| --device | 运行设备。默认为 `gpu`，可选列表：`['cpu', 'gpu', 'huawei_ascend_npu', 'kunlunxin_xpu']`。 |
| --device_id | `gpu` 设备的 id。若 `device_id` 为`-1`，视为使用 `cpu` 推理。 |
| --inference_steps | `UNet` 模型运行的次数，默认为 `50`。 |
| --benchmark_steps | `Benchmark` 运行的次数，默认为 `1`。 |
| --use_fp16 | 是否使用 `fp16` 精度。默认为 `False`。使用 `paddle_tensorrt` 后端时可以设为 `True` 开启。 |
| --task_name | 任务类型，默认为`text2img`，可选列表：`['text2img_control', 'img2img_control', 'inpaint_legacy_control', 'all']`。|
| --scheduler | 采样器类型。默认为 `'preconfig-euler-ancestral'`。可选列表：`['pndm', 'lms', 'preconfig-lms', 'euler', 'euler-ancestral', 'preconfig-euler-ancestral', 'dpm-multi', 'dpm-single', 'unipc-multi', 'ddim', 'ddpm', 'deis-multi', 'heun', 'kdpm2-ancestral', 'kdpm2']`。|
| --width | 生成图片的宽度，取值范围 512~768。|
| --height | 生成图片的高度，取值范围 512~768。|
| --low_threshold | Canny算法最后一步中，小于该阈值的像素直接置为0，默认为 100。 |
| --high_threshold | Canny算法最后一步中，大于该阈值的像素直接置为255，默认为 200。 |
| --is_sd2_0 | 是否为sd2.0的模型？默认为 False 。 |

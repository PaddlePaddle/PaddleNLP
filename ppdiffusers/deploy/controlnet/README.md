# FastDeploy ControlNet 模型高性能部署

 **目录**
   * [部署模型准备](#部署模型准备)
   * [环境依赖](#环境依赖)
   * [快速体验](#快速体验)
       * [ControlNet文图生成（ControlNet-Text-to-Image Generation）](#ControlNet文图生成)
       * [ControlNet文本引导的图像变换（ControlNet-Image-to-Image Text-Guided Generation）](#ControlNet文本引导的图像变换)
       * [ControlNet文本引导的图像编辑（ControlNet-Text-Guided Image Inpainting）](#ControlNet文本引导的图像编辑)

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

<a name="ControlNet文图生成"></a>

### ControlNet文图生成（ControlNet-Text-to-Image Generation）

下面左图是我们提供的初始图片，右图是经过OpenCV中的Canny算法处理后得到的边缘检测图片。

![bird](https://user-images.githubusercontent.com/50394665/225192117-3ec7a61c-227b-4056-a076-d37759f8411b.png)
![control_bird_canny](https://user-images.githubusercontent.com/50394665/225192606-47ba975f-f6cc-4555-8d85-870dc1327b45.png)

> Tips：为了能够跑出最快的推理速度，如果是使用`A卡GPU`的用户，请保证`低于8.5版本的TRT`不在`LD_LIBRARY_PATH`路径上。

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成 `Canny to Image` 任务。

```sh
python infer.py --model_dir ./control_sd15_canny --scheduler "ddim" --backend paddle --task text2img_control
```

脚本的输入提示语句为 **"bird"**， 得到的图像文件为 `text2img_control.png`。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![text2img_control](https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/c2f5e7f0-8abf-4a6c-bc38-bcaf8f58cac5)

如果使用 `paddle_tensorrt` 推理引擎后端及开启`半精度推理`，则可执行以下命令完成推理：

```sh
python infer.py --model_dir control_sd15_canny --scheduler "preconfig-euler-ancestral" --backend paddle_tensorrt --device gpu --benchmark_steps 10 --use_fp16 True --task text2img_control
```

经测试，使用上述命令，在 80G A100 机器上能够跑出 `1.111716 s` 的成绩。

同时，我们还提供基于两阶段 HiresFix 的可控文图生成示例。下面将指定模型目录，指定任务名称为 `hiresfix` 后，运行 `infer.py` 脚本，完成`两阶段hiresfix任务`，在第一阶段我们生成了 `512x512分辨率` 的图片，然后在第二阶段我们在一阶段的基础上修复生成了 `768x768分辨率` 图片。

|       without hiresfix       |       with hiresfix       |
|:-------------------:|:-------------------:|
|![][without-hiresfix]|![][with-hiresfix]|

[without-hiresfix]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/2e3002bc-4a55-4b73-869f-b4e065e62644
[with-hiresfix]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/3f80ce29-8854-4877-911a-11da928a0559

在80G A100上，ppdiffusers==0.16.1、fastdeploy==1.0.7、develop paddle、cuda11.7 的环境下，我们测出了如下的速度。
- without hiresfix 的速度为：Mean latency: 2.715479 s, p50 latency: 2.715581 s, p90 latency: 2.717518 s, p95 latency: 2.719844 s.
- with hiresfix 的速度为：Mean latency: 2.027131 s, p50 latency: 2.026837 s, p90 latency: 2.028943 s, p95 latency: 2.032201 s.

<a name="ControlNet文本引导的图像变换"></a>

### ControlNet文本引导的图像变换（ControlNet-Image-to-Image Text-Guided Generation）

```sh
python infer.py --model_dir ./control_sd15_canny --scheduler "euler-ancestral" --backend paddle_tensorrt --use_fp16 True --device gpu --task_name img2img_control
```

脚本输入的提示语句为 **"A fantasy landscape, trending on artstation"**，运行得到的图像文件为 img2img_control.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       output image       |
|:-------------------:|:-------------------:|
|![][sketch-mountains-input]|![][fantasy_landscape]|

[sketch-mountains-input]: https://user-images.githubusercontent.com/10826371/217207485-09ee54de-4ba2-4cff-9d6c-fd426d4c1831.png
[fantasy_landscape]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/c3727ee3-2955-4ae9-9fbd-a434a9613eda

<a name="ControlNet文本引导的图像编辑"></a>

### ControlNet文本引导的图像编辑（ControlNet-Text-Guided Image Inpainting）

```sh
python infer.py ./control_sd15_canny --scheduler "euler-ancestral" --backend paddle_tensorrt --use_fp16 True --device gpu --task_name inpaint_legacy_control
```

脚本输入的提示语为 **"Face of a yellow cat, high resolution, sitting on a park bench"**，运行得到的图像文件为 inpaint_legacy_control.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       mask image       |       output image
|:-------------------:|:-------------------:|:-------------------:|
|![][input]|![][mask]|![][output]|

[input]: https://user-images.githubusercontent.com/10826371/217423470-b2a3f8ac-618b-41ee-93e2-121bddc9fd36.png
[mask]: https://user-images.githubusercontent.com/10826371/217424068-99d0a97d-dbc3-4126-b80c-6409d2fd7ebc.png
[output]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/63735f7d-038a-48d0-a688-7c1aa4912ab0


#### 参数说明

`infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。默认为 `runwayml/stable-diffusion-v1-5-canny@fastdeploy` |
| --backend | 推理引擎后端。默认为 `paddle_tensorrt`，可选列表：`['onnx_runtime', 'paddle', 'paddlelite', 'paddle_tensorrt', 'tensorrt']`。 |
| --device | 运行设备。默认为 `gpu`，可选列表：`['cpu', 'gpu', 'huawei_ascend_npu', 'kunlunxin_xpu']`。 |
| --device_id | `gpu` 设备的 id。若 `device_id` 为`-1`，视为使用 `cpu` 推理。 |
| --inference_steps | `UNet` 模型运行的次数，默认为 `50`。 |
| --benchmark_steps | `Benchmark` 运行的次数，默认为 `1`。 |
| --use_fp16 | 是否使用 `fp16` 精度。默认为 `False`。使用 `paddle_tensorrt` 后端及 `kunlunxin_xpu` 设备时可以设为 `True` 开启。 |
| --task_name | 任务类型，默认为`text2img`，可选列表：`['text2img_control', 'img2img_control', 'inpaint_legacy_control', 'hiresfix_control', 'all']`。|
| --scheduler | 采样器类型。默认为 `'preconfig-euler-ancestral'`。可选列表：`['pndm', 'lms', 'euler', 'euler-ancestral', 'preconfig-euler-ancestral', 'dpm-multi', 'dpm-single', 'unipc-multi', 'ddim', 'ddpm', 'deis-multi', 'heun', 'kdpm2-ancestral', 'kdpm2']`。|
| --infer_op | 推理所采用的op，可选列表 `['zero_copy_infer', 'raw', 'all']`，`zero_copy_infer`推理速度更快，默认值为`zero_copy_infer`。 |
| --parse_prompt_type | 处理prompt文本所使用的方法，可选列表 `['raw', 'lpw']`，`lpw`可强调句子中的单词，并且支持更长的文本输入，默认值为`lpw`。 |
| --low_threshold | Canny算法最后一步中，小于该阈值的像素直接置为0，默认值为 100。 |
| --high_threshold | Canny算法最后一步中，大于该阈值的像素直接置为255，默认值为 200。 |
| --width | 生成图片的宽度，取值范围 512~768。默认值为 512。|
| --height | 生成图片的高度，取值范围 512~768。默认值为 512。|
| --hr_resize_width | hiresfix 所要生成的宽度，取值范围 512~768。默认值为 768。|
| --hr_resize_height | hiresfix 所要生成的高度，取值范围 512~768。默认值为 768。|
| --is_sd2_0 | 是否为sd2.0的模型？默认为 False 。|

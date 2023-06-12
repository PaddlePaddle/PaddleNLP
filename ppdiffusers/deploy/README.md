# FastDeploy Stable Diffusion 模型高性能部署

 **目录**
   * [部署模型准备](#部署模型准备)
   * [环境依赖](#环境依赖)
   * [快速体验](#快速体验)
       * [文图生成（Text-to-Image Generation）](#文图生成)
       * [文本引导的图像变换（Image-to-Image Text-Guided Generation）](#文本引导的图像变换)
       * [文本引导的图像编辑（Text-Guided Image Inpainting）](#文本引导的图像编辑)

⚡️[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)是一款全场景、易用灵活、极致高效的AI推理部署工具，为开发者提供多硬件、多推理引擎后端的部署能力。开发者只需调用一行代码即可随意切换硬件、推理引擎后端。本示例展现如何通过 FastDeploy 将我们 PPDiffusers 训练好的 Stable Diffusion 模型进行多硬件、多推理引擎后端高性能部署。

<a name="部署模型准备"></a>

## 部署模型准备

本示例需要使用训练模型导出后的部署模型，可参考[模型导出文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/deploy/export.md)导出部署模型。

<a name="环境依赖"></a>

## 环境依赖

在示例中使用了 FastDeploy，需要执行以下命令安装依赖。

```shell
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

<a name="快速体验"></a>

## 快速体验

我们经过部署模型准备，可以开始进行测试。本目录提供 StableDiffusion 模型支持的三种任务，分别是文图生成、文本引导的图像变换以及文本引导的图像编辑。

<a name="文图生成"></a>

### 文图生成（Text-to-Image Generation）


下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文图生成任务。

```
python infer.py --model_dir stable-diffusion-v1-4/ --scheduler "pndm" --backend paddle --task_name text2img
```

脚本的输入提示语句为 **"a photo of an astronaut riding a horse on mars"**， 得到的图像文件为 fd_astronaut_rides_horse.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![fd_astronaut_rides_horse.png](https://user-images.githubusercontent.com/10826371/200261112-68e53389-e0a0-42d1-8c3a-f35faa6627d7.png)

如果使用 stable-diffusion-v1-5 模型，则可执行以下命令完成推理：

```
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler_ancestral" --backend paddle_tensorrt --use_fp16 True --device gpu --task_name text2img
```


<a name="文本引导的图像变换"></a>

### 文本引导的图像变换（Image-to-Image Text-Guided Generation）

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文本引导的图像变换任务。

```
python infer.py --model_dir stable-diffusion-v1-4/ --scheduler "pndm" --backend paddle_tensorrt --use_fp16 True --device gpu --task_name img2img
```

脚本输入的提示语句为 **"A fantasy landscape, trending on artstation"**，运行得到的图像文件为 fantasy_landscape.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       output image       |
|:-------------------:|:-------------------:|
|![][sketch-mountains-input]|![][fantasy_landscape]|

[sketch-mountains-input]: https://user-images.githubusercontent.com/10826371/217207485-09ee54de-4ba2-4cff-9d6c-fd426d4c1831.png
[fantasy_landscape]: https://user-images.githubusercontent.com/10826371/217200795-811a8c73-9fb3-4445-b363-b445c7ee52cd.png



如果使用 stable-diffusion-v1-5 模型，则可执行以下命令完成推理：

```
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler_ancestral" --backend paddle_tensorrt --use_fp16 True --device gpu --task_name img2img
```


同时，我们还提供基于 CycleDiffusion 的文本引导的图像变换示例。下面将指定模型目录，运行 `infer.py` 脚本，完成文本引导的图像变换任务。

```
python infer.py --model_dir stable-diffusion-v1-4/ --backend paddle_tensorrt --use_fp16 True --device gpu --task_name cycle_diffusion
```

脚本输入的源提示语句为 **"An astronaut riding a horse"**，目标提示语句为 **"An astronaut riding an elephant"**，运行得到的图像文件为 horse_to_elephant.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       output image       |
|:-------------------:|:-------------------:|
|![][horse]|![][elephant]|

[horse]: https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/An%20astronaut%20riding%20a%20horse.png
[elephant]: https://user-images.githubusercontent.com/10826371/223315865-4490b586-1de7-4616-a245-9c008c3ffb6b.png

<a name="文本引导的图像编辑"></a>

### 文本引导的图像编辑（Text-Guided Image Inpainting）

注意！当前有两种版本的图像编辑代码，一个是 Legacy 版本，一个是正式版本，下面将分别介绍两种版本的使用示例。

#### Legacy 版本

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文本引导的图像编辑任务。

```
python infer.py --model_dir stable-diffusion-v1-4/ --scheduler euler_ancestral --backend paddle_tensorrt --use_fp16 True --device gpu --task_name inpaint_legacy
```

脚本输入的提示语为 **"Face of a yellow cat, high resolution, sitting on a park bench"**，运行得到的图像文件为 cat_on_bench_new.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       mask image       |       output image
|:-------------------:|:-------------------:|:-------------------:|
|![][input]|![][mask]|![][output]|

[input]: https://user-images.githubusercontent.com/10826371/217423470-b2a3f8ac-618b-41ee-93e2-121bddc9fd36.png
[mask]: https://user-images.githubusercontent.com/10826371/217424068-99d0a97d-dbc3-4126-b80c-6409d2fd7ebc.png
[output]: https://user-images.githubusercontent.com/10826371/217455594-187aa99c-b321-4535-aca0-9159ad658a97.png

如果使用 stable-diffusion-v1-5 模型，则可执行以下命令完成推理：

```
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler euler_ancestral --backend paddle_tensorrt --use_fp16 True --device gpu --task_name inpaint_legacy
```

#### 正式版本

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文本引导的图像编辑任务。

```
python infer.py --model_dir stable-diffusion-v1-5-inpainting/ --scheduler euler_ancestral --backend paddle_tensorrt --use_fp16 True --device gpu --task_name inpaint
```

脚本输入的提示语为 **"Face of a yellow cat, high resolution, sitting on a park bench"**，运行得到的图像文件为 cat_on_bench_new.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       mask image       |       output image
|:-------------------:|:-------------------:|:-------------------:|
|![][input_2]|![][mask_2]|![][output_2]|

[input_2]: https://user-images.githubusercontent.com/10826371/217423470-b2a3f8ac-618b-41ee-93e2-121bddc9fd36.png
[mask_2]: https://user-images.githubusercontent.com/10826371/217424068-99d0a97d-dbc3-4126-b80c-6409d2fd7ebc.png
[output_2]: https://user-images.githubusercontent.com/10826371/217454490-7d6c6a89-fde6-4393-af8e-05e84961b354.png

#### 参数说明

`infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。


| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。默认为 `runwayml/stable-diffusion-v1-5@fastdeploy` |
| --backend | 推理引擎后端。默认为 `paddle_tensorrt`，可选列表：`['onnx_runtime', 'paddle', 'paddlelite', 'paddle_tensorrt']`。 |
| --device | 运行设备。默认为 `cpu`，可选列表：`['cpu', 'gpu', 'huawei_ascend_npu', 'kunlunxin_xpu']`。 |
| --device_id | `gpu` 设备的 id。若 `device_id` 为`-1`，视为使用 `cpu` 推理。 |
| --inference_steps | `UNet` 模型运行的次数，默认为 `50`。 |
| --benchmark_steps | `Benchmark` 运行的次数，默认为 `1`。 |
| --use_fp16 | 是否使用 `fp16` 精度。默认为 `False`。使用 `paddle_tensorrt` 后端时可以设为 `True` 开启。 |
| --task_name | 任务类型，默认为`text2img`，可选列表：`['text2img', 'img2img', 'inpaint', 'inpaint_legacy', 'cycle_diffusion', 'all']`。 注意，当`task_name`为`inpaint`时候，我们需要配合`runwayml/stable-diffusion-inpainting@fastdeploy`权重才能正常使用。|
| --scheduler | 采样器类型。默认为 `'pndm'`。可选列表：`['pndm', 'lms', 'preconfig-lms', 'euler', 'euler-ancestral', 'preconfig-euler-ancestral', 'dpm-multi', 'dpm-single', 'unipc-multi', 'ddim', 'ddpm', 'deis-multi', 'heun', 'kdpm2-ancestral', 'kdpm2']`。|
| --width | 生成图片的宽度，取值范围 512~768。|
| --height | 生成图片的高度，取值范围 512~768。|

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


下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `text_to_img_infer.py` 脚本，完成文图生成任务。

```
python text_to_img_infer.py --model_dir stable-diffusion-v1-4/ --scheduler "pndm" --backend paddle
```

脚本的输入提示语句为 **"a photo of an astronaut riding a horse on mars"**， 得到的图像文件为 fd_astronaut_rides_horse.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![fd_astronaut_rides_horse.png](https://user-images.githubusercontent.com/10826371/200261112-68e53389-e0a0-42d1-8c3a-f35faa6627d7.png)

如果使用 stable-diffusion-v1-5 模型，则可执行以下命令完成推理：

```
python text_to_img_infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler_ancestral" --backend paddle_tensorrt --use_fp16 True --device gpu
```

#### 参数说明

`text_to_img_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。

<details><summary>&emsp; 命令行参数说明 </summary>

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。 |
| --model_format | 模型格式。默认为 `'paddle'`，可选列表：`['paddle', 'onnx']`。 |
| --backend | 推理引擎后端。默认为 `paddle`，可选列表：`['onnx_runtime', 'paddle', 'paddlelite', 'paddle_tensorrt']`，当模型格式为 `onnx` 时，可选列表为 `['onnx_runtime']`。 |
| --device | 运行设备。默认为 `cpu`，可选列表：`['cpu', 'gpu', 'huawei_ascend_npu', 'kunlunxin_xpu']`。 |
| --scheduler | StableDiffusion 模型的 scheduler。默认为 `'pndm'`。可选列表：`['pndm', 'euler_ancestral']`。|
| --unet_model_prefix | UNet 模型前缀。默认为 `unet`。 |
| --vae_model_prefix | VAE 模型前缀。默认为 `vae_decoder`。 |
| --text_encoder_model_prefix | TextEncoder 模型前缀。默认为 `text_encoder`。 |
| --inference_steps | UNet 模型运行的次数，默认为 50。 |
| --image_path | 生成图片的路径。默认为 `fd_astronaut_rides_horse.png`。  |
| --device_id | gpu 设备的 id。若 `device_id` 为-1，视为使用 cpu 推理。 |
| --use_fp16 | 是否使用 fp16 精度。默认为 `False`。使用 tensorrt 或者 paddle_tensorrt 后端时可以设为 `True` 开启。 |

</details>

<a name="文本引导的图像变换"></a>

### 文本引导的图像变换（Image-to-Image Text-Guided Generation）

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `img_to_img_infer.py` 脚本，完成文本引导的图像变换任务。

```
python img_to_img_infer.py --model_dir stable-diffusion-v1-4/ --scheduler "pndm" --backend paddle_tensorrt --use_fp16 True --device gpu
```

脚本输入的提示语句为 **"A fantasy landscape, trending on artstation"**，待变换的图像为：

![sketch-mountains-input.png](https://user-images.githubusercontent.com/10826371/217207485-09ee54de-4ba2-4cff-9d6c-fd426d4c1831.png)


运行得到的图像文件为 fantasy_landscape.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![fantasy_landscape.png](https://user-images.githubusercontent.com/10826371/217200795-811a8c73-9fb3-4445-b363-b445c7ee52cd.png)


如果使用 stable-diffusion-v1-5 模型，则可执行以下命令完成推理：

```
python img_to_img_infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler_ancestral" --backend paddle_tensorrt --use_fp16 True --device gpu
```

#### 参数说明

`img_to_img_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。

<details><summary>&emsp; 命令行参数说明 </summary>

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。 |
| --model_format | 模型格式。默认为 `'paddle'`，可选列表：`['paddle', 'onnx']`。 |
| --backend | 推理引擎后端。默认为 `paddle`，可选列表：`['onnx_runtime', 'paddle', 'paddlelite', 'paddle_tensorrt']`，当模型格式为 `onnx` 时，可选列表为 `['onnx_runtime']`。 |
| --device | 运行设备。默认为 `cpu`，可选列表：`['cpu', 'gpu', 'huawei_ascend_npu', 'kunlunxin_xpu']`。 |
| --scheduler | StableDiffusion 模型的 scheduler。默认为 `'pndm'`。可选列表：`['pndm', 'euler_ancestral']`。|
| --unet_model_prefix | UNet 模型前缀。默认为 `unet`。 |
| --vae_model_prefix | VAE 模型前缀。默认为 `vae_decoder`。 |
| --text_encoder_model_prefix | TextEncoder 模型前缀。默认为 `text_encoder`。 |
| --inference_steps | UNet 模型运行的次数，默认为 50。 |
| --image_path | 生成图片的路径。默认为 `fantasy_landscape.png`。  |
| --device_id | gpu 设备的 id。若 `device_id` 为-1，视为使用 cpu 推理。 |
| --use_fp16 | 是否使用 fp16 精度。默认为 `False`。使用 tensorrt 或者 paddle_tensorrt 后端时可以设为 `True` 开启。 |

</details>

同时，我们还提供基于 CycleDiffusion 的文本引导的图像变换示例。下面将指定模型目录，运行 `text_guided_img_to_img_infer.py` 脚本，完成文本引导的图像变换任务。

```
python text_guided_img_to_img_infer.py --model_dir stable-diffusion-v1-4/ --backend paddle_tensorrt --use_fp16 True --device gpu
```

脚本输入的源提示语句为 **"An astronaut riding a horse"**，目标提示语句为 **"An astronaut riding an elephant"**， 待变换的图像为：

![horse](https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/An%20astronaut%20riding%20a%20horse.png)

运行得到的图像文件为 horse_to_elephant.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![image](https://user-images.githubusercontent.com/10826371/223315865-4490b586-1de7-4616-a245-9c008c3ffb6b.png)

<a name="文本引导的图像编辑"></a>

### 文本引导的图像编辑（Text-Guided Image Inpainting）

注意！当前有两种版本的图像编辑代码，一个是 Legacy 版本，一个是正式版本，下面将分别介绍两种版本的使用示例。

#### Legacy 版本

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `inpaint_legacy_infer.py` 脚本，完成文本引导的图像编辑任务。

```
python inpaint_legacy_infer.py --model_dir stable-diffusion-v1-4/ --scheduler euler_ancestral --backend paddle_tensorrt --use_fp16 True --device gpu
```

脚本输入的提示语为 **"Face of a yellow cat, high resolution, sitting on a park bench"**，待变换的图像为：

![image](https://user-images.githubusercontent.com/10826371/217423470-b2a3f8ac-618b-41ee-93e2-121bddc9fd36.png)

mask 图像为：

![overture-creations-mask](https://user-images.githubusercontent.com/10826371/217424068-99d0a97d-dbc3-4126-b80c-6409d2fd7ebc.png)


运行得到的图像文件为 cat_on_bench_new.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![image](https://user-images.githubusercontent.com/10826371/217455594-187aa99c-b321-4535-aca0-9159ad658a97.png)

如果使用 stable-diffusion-v1-5 模型，则可执行以下命令完成推理：

```
python inpaint_legacy_infer.py --model_dir stable-diffusion-v1-5/ --scheduler euler_ancestral --backend paddle_tensorrt --use_fp16 True --device gpu
```

#### 正式版本

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `inpaint_infer.py` 脚本，完成文本引导的图像编辑任务。

```
python inpaint_infer.py --model_dir stable-diffusion-v1-5-inpainting/ --scheduler euler_ancestral --backend paddle_tensorrt --use_fp16 True --device gpu
```

脚本输入的提示语为 **"Face of a yellow cat, high resolution, sitting on a park bench"**，待变换的图像为：

![image](https://user-images.githubusercontent.com/10826371/217423470-b2a3f8ac-618b-41ee-93e2-121bddc9fd36.png)

mask 图像为：

![overture-creations-mask](https://user-images.githubusercontent.com/10826371/217424068-99d0a97d-dbc3-4126-b80c-6409d2fd7ebc.png)


运行得到的图像文件为 cat_on_bench_new.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![image](https://user-images.githubusercontent.com/10826371/217454490-7d6c6a89-fde6-4393-af8e-05e84961b354.png)


#### 参数说明

`inpaint_legacy_infer.py` 和 `inpaint_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。

<details><summary>&emsp; 命令行参数说明 </summary>

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。 |
| --model_format | 模型格式。默认为 `'paddle'`，可选列表：`['paddle', 'onnx']`。 |
| --backend | 推理引擎后端。默认为 `paddle`，可选列表：`['onnx_runtime', 'paddle', 'paddlelite', 'paddle_tensorrt']`，当模型格式为 `onnx` 时，可选列表为 `['onnx_runtime']`。 |
| --device | 运行设备。默认为 `cpu`，可选列表：`['cpu', 'gpu', 'huawei_ascend_npu', 'kunlunxin_xpu']`。 |
| --scheduler | StableDiffusion 模型的 scheduler。默认为 `'pndm'`。可选列表：`['pndm', 'euler_ancestral']`。|
| --unet_model_prefix | UNet 模型前缀。默认为 `unet`。 |
| --vae_decoder_model_prefix | VAE Decoder 模型前缀。默认为 `vae_decoder`。 |
| --vae_encoder_model_prefix | VAE Encoder 模型前缀。默认为 `vae_encoder`。 |
| --text_encoder_model_prefix | TextEncoder 模型前缀。默认为 `text_encoder`。 |
| --inference_steps | UNet 模型运行的次数，默认为 50。 |
| --image_path | 生成图片的路径。默认为 `cat_on_bench_new.png`。  |
| --device_id | gpu 设备的 id。若 `device_id` 为-1，视为使用 cpu 推理。 |
| --use_fp16 | 是否使用 fp16 精度。默认为 `False`。使用 tensorrt 或者 paddle_tensorrt 后端时可以设为 `True` 开启。 |

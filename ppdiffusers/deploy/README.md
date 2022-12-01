# FastDeploy Diffusion模型高性能部署

本示例展现如何通过FastDeploy将我们PPDiffusers训练好的Diffusion模型进行多推理引擎后端高性能部署。

### 部署模型准备

本示例需要使用训练模型导出后的部署模型。有两种部署模型的获取方式：

- 模型导出方式，可参考[模型导出文档](./export.md)导出部署模型。
- 下载部署模型。为了方便开发者快速测试本示例，我们已经将部分`Diffusion`模型预先导出，开发者只要下载模型就可以快速测试:

| 模型 | Scheduler |
|----------|--------------|
| [CompVis/stable-diffusion-v1-4](https://bj.bcebos.com/fastdeploy/models/stable-diffusion/CompVis/stable-diffusion-v1-4.tgz) | PNDM |
| [runwayml/stable-diffusion-v1-5](https://bj.bcebos.com/fastdeploy/models/stable-diffusion/runwayml/stable-diffusion-v1-5.tgz) | EulerAncestral |

## 环境依赖

在示例中使用了FastDeploy，需要执行以下命令安装依赖。

```shell
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

### 快速体验

我们经过部署模型准备，可以开始进行测试。下面将指定模型目录以及推理引擎后端，运行`infer.py`脚本，完成推理。

```
python infer.py --model_dir stable-diffusion-v1-4/ --scheduler "pndm" --backend paddle
```

得到的图像文件为fd_astronaut_rides_horse.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![fd_astronaut_rides_horse.png](https://user-images.githubusercontent.com/10826371/200261112-68e53389-e0a0-42d1-8c3a-f35faa6627d7.png)

如果使用stable-diffusion-v1-5模型，则可执行以下命令完成推理：

```
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler_ancestral" --backend paddle
```

#### 参数说明

`infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。 |
| --model_format | 模型格式。默认为`'paddle'`，可选列表：`['paddle', 'onnx']`。 |
| --backend | 推理引擎后端。默认为`paddle`，可选列表：`['onnx_runtime', 'paddle']`，当模型格式为`onnx`时，可选列表为`['onnx_runtime']`。 |
| --scheduler | StableDiffusion 模型的scheduler。默认为`'pndm'`。可选列表：`['pndm', 'euler_ancestral']`，StableDiffusio模型对应的scheduler可参考[ppdiffuser模型列表](../examples/textual_inversion)。|
| --unet_model_prefix | UNet模型前缀。默认为`unet`。 |
| --vae_model_prefix | VAE模型前缀。默认为`vae_decoder`。 |
| --text_encoder_model_prefix | TextEncoder模型前缀。默认为`text_encoder`。 |
| --inference_steps | UNet模型运行的次数，默认为100。 |
| --image_path | 生成图片的路径。默认为`fd_astronaut_rides_horse.png`。  |
| --device_id | gpu设备的id。若`device_id`为-1，视为使用cpu推理。 |
| --use_fp16 | 是否使用fp16精度。默认为`False`。使用tensorrt或者paddle-tensorrt后端时可以设为`True`开启。 |

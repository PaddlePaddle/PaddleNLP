# ControlNet
[ControlNet](https://arxiv.org/abs/2302.05543) 是一种通过添加额外条件来控制扩散模型的神经网络结构。
<p align="center">
    <img src="https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/he.png">
</p>

## 安装依赖
在运行这部分代码前，我们需要安装develop分支的ppdiffusers库：
```bash
cd ppdiffusers
python setup.py install
```
此外我们还需要安装相关依赖：
```bash
pip install -r requirements.txt
```


# ControlNet with Stable Diffusion预训练模型
除文本提示外，ControlNet还需要一个控制图作为控制条件。每个预训练模型使用不同的控制方法进行训练，其中每种方法对应一种不同的控制图。例如，Canny to Image要求控制图像是Canny边缘检测的输出图像，而Pose to Image要求控制图是OpenPose骨骼姿态检测图像。目前我们支持如下控制方式及预训练模型。
## Canny to Image
采用Canny边缘检测图片作为控制条件。
```
python gradio_canny2image.py
```
![image](https://user-images.githubusercontent.com/20476674/222131385-0dfaa370-fb11-4b2b-9ef5-36143557578b.png)

## Hed to Image
采用Hed边缘检测图片作为控制条件。
```
python gradio_hed2image.py
```
![image](https://user-images.githubusercontent.com/20476674/223642261-d5bdbd83-06f9-459b-8224-486f2235f7a6.png)


## Pose to Image
采用OpenPose姿态图片作为控制条件。
```
python gradio_pose2image.py
```
![image](https://user-images.githubusercontent.com/20476674/222131475-4dc8582a-d2a2-447a-9724-85461de04c26.png)
## Semantic Segmentation to Image
采用ADE20K分割协议的图片作为控制条件。
```
python gradio_seg2image_segmenter.py
```
![image](https://user-images.githubusercontent.com/20476674/222131908-b0c52512-ef42-4e4b-8fde-62c12c600ff2.png)
# ControlNet模型训练

## Fill50K 训练例子

作为案例，我们将使用 Fill50K 数据集，带领大家训练 ControlNet 模型。首先我们需要下载数据集。
```sh
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/training/fill50k.zip
unzip -o fill50k.zip
```
注意：下面的代码需要在32G V100上才可以正常运行。

### 单机单卡训练
```bash
export FLAGS_conv_workspace_size_limit=4096
python -u train_txt2img_control_trainer.py \
    --do_train \
    --output_dir ./sd15_control \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --sd_locked True \
    --max_steps 10000000 \
    --logging_steps 50 \
    --image_logging_steps 400 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_path ./fill50k \
    --recompute True \
    --overwrite_output_dir
```

`train_txt2img_control_trainer.py`代码可传入的参数解释如下：
> * `--vae_name_or_path`: 预训练`vae`模型名称或地址，`runwayml/stable-diffusion-v1-5/vae`，程序将自动从BOS上下载预训练好的权重。
> * `--text_encoder_name_or_path`: 预训练`text_encoder`模型名称或地址，`runwayml/stable-diffusion-v1-5/text_encoder`，程序将自动从BOS上下载预训练好的权重。
> * `--unet_name_or_path`: 预训练`unet`模型名称或地址，`runwayml/stable-diffusion-v1-5/unet`，程序将自动从BOS上下载预训练好的权重。
> * `--pretrained_model_name_or_path`: 加载预训练模型的名称或本地路径，如`runwayml/stable-diffusion-v1-5`，`pretrained_model_name_or_path`的优先级高于`vae_name_or_path`, `text_encoder_name_or_path`和`unet_name_or_path`。
> * `--per_device_train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的step中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的batch_size。
> * `--learning_rate`: 学习率。
> * `--weight_decay`: `AdamW`优化器的`weight_decay`。
> * `--max_steps`: 最大的训练步数。
> * `--save_steps`: 每间隔多少步`（global step步数）`，保存模型。
> * `--save_total_limit`: 最多保存多少个模型。
> * `--lr_scheduler_type`: 要使用的学习率调度策略。默认为 `constant`。
> * `--warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
> * `--image_logging_steps`: 每隔多少步，log训练过程中的图片，默认为`1000`步，注意`image_logging_steps`需要是`logging_steps`的整数倍。
> * `--logging_steps`: logging日志的步数，默认为`50`步。
> * `--output_dir`: 模型保存路径。
> * `--seed`: 随机种子，为了可以复现训练结果，Tips：当前paddle设置该随机种子后仍无法完美复现。
> * `--dataloader_num_workers`: Dataloader所使用的`num_workers`参数。
> * `--file_path`: 训练数据文件夹所在的地址，上述例子我们使用了`fill50k`目录。
> * `--num_inference_steps`: 推理预测时候使用的步数。
> * `--model_max_length`: `tokenizer`中的`model_max_length`参数，超过该长度将会被截断。
> * `--tokenizer_name`: 我们需要使用的`tokenizer_name`，我们可以使用英文的分词器`bert-base-uncased`，也可以使用中文的分词器`ernie-1.0`。
> * `--use_ema`: 是否对`unet`使用`ema`，默认为`False`。
> * `--max_grad_norm`: 梯度剪裁的最大norm值，`-1`表示不使用梯度裁剪策略。
> * `--use_paddle_conv_init`: 是否使用`paddle`的卷积初始化策略，当我们开启该策略后可以很快发现在`fill50k`数据集上，模型很快就收敛了，默认值为 `False`。
> * `--recompute`: 是否开启重计算，(`bool`, 可选, 默认为 `False`)，在开启后我们可以增大`batch_size`。
> * `--fp16`: 是否使用 fp16 混合精度训练而不是 fp32 训练。(`bool`, 可选, 默认为 `False`)
> * `--fp16_opt_level`: 混合精度训练模式，可为``O1``或``O2``模式，默认``O1``模式，默认O1. 只在fp16选项开启时候生效。
> * `--is_ldmbert`: 是否使用`ldmbert`作为`text_encoder`，默认为`False`，即使用 `clip text_encoder`。



**Tips**:
> 结合 `paddle` 文档和 `torch` 文档可知，`paddle` 卷积层初始化是 `Xavier Normal`，`torch` 卷积层初始化是 `Uniform`，初始化方法边界值是`(-sqrt(groups/(in_channels*prod(*kernal_size))), sqrt(groups/(in_channels*prod(*kernal_size))))`。
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/222323097-1ff4472c-b4d0-48a0-92c7-44fbb18997f5.png" width="700">
    <img src="https://user-images.githubusercontent.com/50394665/222323163-11ecf153-1f79-4384-b455-d5429748d184.png" width="700">
</p>

### 单机多卡训练 (多机多卡训练，仅需在 paddle.distributed.launch 后加个 --ips IP1,IP2,IP3,IP4)
```bash
export FLAGS_conv_workspace_size_limit=4096
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_control_trainer.py \
    --do_train \
    --output_dir ./sd15_control \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --sd_locked True \
    --max_steps 10000000 \
    --logging_steps 50 \
    --image_logging_steps 400 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_path ./fill50k \
    --recompute True \
    --overwrite_output_dir
```

## 模型推理
待模型训练完毕，会在`output_dir`保存训练好的模型权重，我们可以使用如下的代码进行推理
```python
from ppdiffusers import StableDiffusionControlNetPipeline, ControlNetModel
from ppdiffusers.utils import load_image
controlnet = ControlNetModel.from_pretrained("./sd15_control/checkpoint-12000/controlnet")
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet = controlnet, safety_checker=None)
canny_edged_image = load_image("https://user-images.githubusercontent.com/50394665/221844474-fd539851-7649-470e-bded-4d174271cc7f.png")
img = pipe(prompt="pale golden rod circle with old lace background", image=canny_edged_image, guidance_scale=9, num_inference_steps=50).images[0]
img.save("demo.png")
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/221844474-fd539851-7649-470e-bded-4d174271cc7f.png">
    <img src="https://user-images.githubusercontent.com/50394665/222058833-7e94bfa5-7cc2-4b9e-9022-37c9d47398de.png">
</p>


# 参考资料
- https://github.com/lllyasviel/ControlNet/edit/main/docs/train.md
- https://github.com/huggingface/diffusers

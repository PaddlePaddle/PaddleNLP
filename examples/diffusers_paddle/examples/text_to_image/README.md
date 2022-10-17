# 微调Stable Diffusion模型

`train_text_to_image.py`脚本展示如何在自定义数据集上微调Stable Diffusion模型。

___Note___:

___该训练代码是实验性质的。由于这里的代码微调了整个`UNet模型`，通常该模型可能会产生过拟合的现象，可能会产生像`"catastrophic forgetting"`的问题。如果用户在自己的数据集上进行微调训练，为了得到更好的训练结果，建议尝试使用不同的参数值。___


## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。

```bash
# 进入examples/diffusers_paddle文件夹，安装diffusers_paddle
pip install -e .
# 安装其他所需的依赖
pip install paddlenlp>=2.4.1 ftfy regex Pillow
```

### 1.2 Pokemon训练教程

为了下载`CompVis/stable-diffusion-v1-4`模型权重，我们需要阅读并签署相关的License。在这里我们默认用户已经阅读并签署了解了相关License，有关License及模型的详细介绍，请访问[CompVis/stable-diffusion-v1-4 card](https://huggingface.co/CompVis/stable-diffusion-v1-4)。

> License: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which our license is based.
<br>

#### 1.2.1 硬件要求
当我们开启`gradient_checkpointing`功能后（Tips：该功能可以在一定程度上减少显存消耗），我们可以在24GB显存的GPU上微调模型。如果想要使用更大的`batch_size`进行更快的训练，建议用户使用具有30GB+显存的显卡。

#### 1.2.2 单机单卡训练
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/pokemon-blip-captions"

python -u train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model"
```
或
```bash
bash run_single.sh
```

#### 1.2.3 单机多卡训练
通过设置`--gpus`，我们可以指定 GPU 为 `0,1,2,3` 卡。这里我们只训练了`4000step`，因为这里的`4000 step x 4卡`近似于`单卡训练 16000 step`。

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/pokemon-blip-captions"

python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=4000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model"
```
或
```bash
bash run_multi.sh
```

#### 1.2.4 预测生成图片
当训练完成后，模型将自动保存到`output_dir`目录，在上述例子中，我们的模型最终保存到了`sd-pokemon-model`文件夹。我们可以使用`StableDiffusionPipeline`快速加载该模型。

```python
from diffusers_paddle import StableDiffusionPipeline

# 我们所需加载的模型地址，这里我们输入了训练时候使用的 output_dir 地址
model_path = "sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

image = pipe(prompt="yoda").images[0]
# 保存图片，我们可以查看 yoda-pokemon.png 图片。
image.save("yoda-pokemon.png")
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/196165976-a999bf68-382c-484d-b86e-5006a05c90d8.png">
</p>

### 1.3 自定义数据集训练教程
如果用户想要在自己的数据集上进行训练，那么需要根据`huggingface的 datasets 库`所需的格式准备数据集，有关数据集的介绍可以查看 [HF dataset的文档](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder-with-metadata).

如果用户想要修改代码中的部分训练逻辑，那么需要修改训练代码。

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# 这里需要输入你自己的训练集路径
export TRAIN_DIR="path_to_your_dataset"

python -u train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-custom-model"
```

## 2 参考资料
- https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
- https://github.com/CompVis/stable-diffusion
- https://huggingface.co/lambdalabs/sd-pokemon-diffusers

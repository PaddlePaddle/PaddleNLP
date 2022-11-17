# 微调Stable Diffusion模型

`train_text_to_image.py`脚本展示如何在自定义数据集上微调Stable Diffusion模型。

___Note___:

___该训练代码是实验性质的。由于这里的代码微调了整个`UNet模型`，通常该模型可能会产生过拟合的现象，可能会产生像`"catastrophic forgetting"`的问题。如果用户在自己的数据集上进行微调训练，为了得到更好的训练结果，建议尝试使用不同的参数值。___


## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。

```bash
pip install -U ppdiffusers visualdl
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
| ppdiffusers支持的模型名称    | huggingface对应的模型地址                           | Tips备注                                                     |
| ---------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| CompVis/stable-diffusion-v1-4            | https://huggingface.co/CompVis/stable-diffusion-v1-4       | 原版SD模型，模型使用PNDM scheduler。                 |
| hakurei/waifu-diffusion                  | https://huggingface.co/hakurei/waifu-diffusion             | Waifu v1-2的模型，模型使用了DDIM scheduler。         |
| hakurei/waifu-diffusion-v1-3             | https://huggingface.co/hakurei/waifu-diffusion             | Waifu v1-3的模型，模型使用了PNDM scheduler。         |
| naclbit/trinart_stable_diffusion_v2_60k  | https://huggingface.co/naclbit/trinart_stable_diffusion_v2 | trinart 经过60k步数训练得到的模型，模型使用了DDIM scheduler。 |
| naclbit/trinart_stable_diffusion_v2_95k  | https://huggingface.co/naclbit/trinart_stable_diffusion_v2 | trinart 经过95k步数训练得到的模型，模型使用了DDIM scheduler。 |
| naclbit/trinart_stable_diffusion_v2_115k | https://huggingface.co/naclbit/trinart_stable_diffusion_v2 | trinart 经过115k步数训练得到的模型，模型使用了DDIM scheduler。 |
| Deltaadams/Hentai-Diffusion              | https://huggingface.co/Deltaadams/Hentai-Diffusion         | Hentai模型，模型使用了PNDM scheduler。                |
| IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1              | https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1         | 中文StableDiffusion模型，模型使用了PNDM scheduler。                |
| IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1              | https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1         | 中文+英文双语言的StableDiffusion模型，模型使用了PNDM scheduler。                |


`train_text_to_image.py`代码可传入的参数解释如下：
> 主要修改的参数
> * `--pretrained_model_name_or_path`: 所使用的Stable Diffusion模型权重名称或者本地下载的模型路径，目前支持了上表中的8种模型权重，我们可直接替换使用。
> * `--dataset_name`: 数据集名字，可填写`HuggingFace hub`已有的数据集名字。
> * `--dataset_config_name`: 数据集所使用的config配置名字。
> * `--train_data_dir`: 如果选择自定义数据集的话，需要提供数据集地址，该地址要遵循 https://huggingface.co/docs/datasets/image_dataset#imagefolder 上的格式。
> * `--image_column`: 图片所在的列名，默认为`image`。
> * `--caption_column`: 文本描述所在的列名，默认为`text`。
> * `--gradient_checkpointing`: 是否开启`gradient_checkpointing`功能，在一定程度上能够更显显存，但是会减慢训练速度。
> * `--use_ema`: 是否使用EMA模型。
> * `--num_train_epochs`: 训练的轮数，默认值为`100`。
> * `--max_train_steps`: 最大的训练步数，当我们设置这个值后，它会重新计算所需的`num_train_epochs`轮数。
> * `--save_steps`: 每间隔多少步`（global step步数）`，保存学习到的文件`learned_embeds.pdparams`。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的step中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的batch_size。

> 可以修改的参数
> * `--learning_rate`: 学习率。
> * `--scale_lr`: 是否根据GPU数量，梯度累积步数，以及批量数对学习率进行缩放。缩放公式：`learning_rate * gradient_accumulation_steps * train_batch_size * num_processes`。
> * `--lr_scheduler`: 要使用的学习率调度策略。默认为 `constant`。
> * `--lr_warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
> * `--train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--center_crop`: 在调整图片宽和高之前是否将裁剪图像居中，默认值为`False`。
> * `--resolution`: 输入给模型的图片`像素大小`，由于用户输入的并不是固定大小的图片，因此代码中会将原始大小的图片压缩成`高度为resolution`，`宽度为resolution`的图片，默认值为`512`。
> * `--output_dir`: 模型训练完所保存的路径，默认设置为`sd-pokemon-model`文件夹，建议用户每训练一个模型可以修改一下输出路径，防止先前已有的模型被覆盖了。

> 基本无需修改的参数
> * `--seed`: 随机种子，为了可以复现训练结果，Tips：当前paddle设置该随机种子后仍无法完美复现。
> * `--adam_beta1`: AdamW 优化器时的 beta1 超参数。默认为 `0.9`。
> * `--adam_beta2`: AdamW 优化器时的 beta2 超参数。默认为 `0.999`。
> * `--adam_weight_decay`: AdamW 优化器时的 weight_decay 超参数。 默认为`0.02`。
> * `--adam_weight_decay`: AdamW 优化器时的 epsilon 超参数。默认为 1e-8。
> * `--max_grad_norm`: 最大梯度范数（用于梯度裁剪）。默认为 `None`表示不使用。
> * `--logging_dir`: Tensorboard 或 VisualDL 记录日志的地址，注意：该地址会与输出目录进行拼接，即，最终的日志地址为`<output_dir>/<logging_dir>`。
> * `--writer_type`: 用于记录日志的工具，可选`["tensorboard", "visualdl"]`，默认为`visualdl`，如果选用`tensorboard`，请使用命令安装`pip install tensorboardX`。


#### 1.2.3 单机多卡训练
通过设置`--gpus`，我们可以指定 GPU 为 `0,1,2,3` 卡。这里我们只训练了`4000step`，因为这里的`4000 step x 4卡`近似于`单卡训练 16000 step`。

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/pokemon-blip-captions"

python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
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

```
├── run_multi.sh # 4卡训练命令
├── run_single.sh # 单卡训练命令
├── train_text_to_image.py # 训练脚本
├── sd-pokemon-model  # 我们指定的输出文件路径
    ├── vae # vae权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── text_encoder # text_encoder权重文件夹
        ├── model_config.json
        ├── model_state.pdparams
    ├── unet # unet权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── scheduler # scheduler文件夹
        ├── scheduler_config.json
    ├── feature_extractor # feature_extractor文件夹
        ├── preprocessor_config.json
    ├── tokenizer # tokenizer文件夹
        ├── tokenizer_config.json
        ├── merges.txt
        ├── special_tokens_map.json
        ├── added_tokens.json
        ├── vocab.json
```

```python
from ppdiffusers import StableDiffusionPipeline

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

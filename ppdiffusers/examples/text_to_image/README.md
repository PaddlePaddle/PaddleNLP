# 微调Stable Diffusion模型

`train_text_to_image.py`脚本展示如何在自定义数据集上微调Stable Diffusion模型。

___Note___:

___该训练代码是实验性质的。由于这里的代码微调了整个`UNet模型`，通常该模型可能会产生过拟合的现象，可能会产生像`"catastrophic forgetting"`的问题。如果用户在自己的数据集上进行微调训练，为了得到更好的训练结果，建议尝试使用不同的参数值。___


## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。

```bash
pip install -U ppdiffusers visualdl -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
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
> * `--height`: 输入给模型的图片`高度`，由于用户输入的并不是固定大小的图片，因此代码中会将原始大小的图片压缩成指定`高度`的图片，默认值为`None`。
> * `--width`: 输入给模型的图片`宽度`，由于用户输入的并不是固定大小的图片，因此代码中会将原始大小的图片压缩成指定`宽度`的图片，默认值为`None`。
> * `--resolution`: 输入给模型图片的`分辨率`，当`高度`或`宽度`为`None`时，我们将会使用`resolution`，默认值为`512`。
> * `--gradient_checkpointing`: 是否开启`gradient_checkpointing`功能，在一定程度上能够更显显存，但是会减慢训练速度。
> * `--output_dir`: 模型训练完所保存的路径，默认设置为`sd-pokemon-model`文件夹，建议用户每训练一个模型可以修改一下输出路径，防止先前已有的模型被覆盖了。
> * `--enable_xformers_memory_efficient_attention`: 是否开启`xformers`，开启后训练速度会变慢，但是能够节省显存。注意我们需要安装develop版本的paddlepaddle！

> 基本无需修改的参数
> * `--seed`: 随机种子，为了可以复现训练结果，Tips：当前paddle设置该随机种子后仍无法完美复现。
> * `--adam_beta1`: `AdamW` 优化器时的 `beta1` 超参数。默认为 `0.9`。
> * `--adam_beta2`: `AdamW` 优化器时的 `beta2` 超参数。默认为 `0.999`。
> * `--adam_weight_decay`: `AdamW` 优化器时的 `weight_decay` 超参数。 默认为`0.02`。
> * `--adam_weight_decay`: `AdamW` 优化器时的 `epsilon` 超参数。默认为 `1e-8`。
> * `--max_grad_norm`: 最大梯度范数（用于梯度裁剪）。默认为 `-1` 表示不使用。
> * `--logging_dir`: Tensorboard 或 VisualDL 记录日志的地址，注意：该地址会与输出目录进行拼接，即，最终的日志地址为`<output_dir>/<logging_dir>`。
> * `--report_to`: 用于记录日志的工具，可选`["tensorboard", "visualdl"]`，默认为`visualdl`，如果选用`tensorboard`，请使用命令安装`pip install tensorboardX`。
> * `--push_to_hub`: 是否将模型上传到 `huggingface hub`，默认值为 `False`。
> * `--hub_token`: 上传到 `huggingface hub` 所需要使用的 `token`，如果我们已经登录了，那么我们就无需填写。
> * `--hub_model_id`: 上传到 `huggingface hub` 的模型库名称， 如果为 `None` 的话表示我们将使用 `output_dir` 的名称作为模型库名称。

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


#### 1.2.4 预测生成图片

当训练完成后，模型将自动保存到`output_dir`目录，在上述例子中，我们的模型最终保存到了`sd-pokemon-model`文件夹。我们可以使用`StableDiffusionPipeline`快速加载该模型。

```
├── train_text_to_image.py # 训练脚本
├── sd-pokemon-model  # 我们指定的输出文件路径
    ├── vae # vae权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── text_encoder # text_encoder权重文件夹
        ├── config.json
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

# 使用 LoRA 和 Text-to-Image 技术进行模型训练

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 是微软研究员引入的一项新技术，主要用于处理大模型微调的问题。目前超过数十亿以上参数的具有强能力的大模型 (例如 GPT-3) 通常在为了适应其下游任务的微调中会呈现出巨大开销。LoRA 建议冻结预训练模型的权重并在每个 Transformer 块中注入可训练层 (秩-分解矩阵)。因为不需要为大多数模型权重计算梯度，所以大大减少了需要训练参数的数量并且降低了 GPU 的内存要求。研究人员发现，通过聚焦大模型的 Transformer 注意力块，使用 LoRA 进行的微调质量与全模型微调相当，同时速度更快且需要更少的计算。

简而言之，LoRA允许通过向现有权重添加一对秩分解矩阵，并只训练这些新添加的权重来适应预训练的模型。这有几个优点：

- 保持预训练的权重不变，这样模型就不容易出现灾难性遗忘 [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114)；
- 秩分解矩阵的参数比原始模型少得多，这意味着训练的 LoRA 权重很容易移植；
- LoRA 注意力层允许通过一个 `scale` 参数来控制模型适应新训练图像的程度。

[cloneofsimo](https://github.com/cloneofsimo) 是第一个在 [LoRA GitHub](https://github.com/cloneofsimo/lora) 仓库中尝试使用 LoRA 训练 Stable Diffusion 的人。

## 训练

**___Note: 如果我们使用 [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 进行训练，那么我们需要将 `resolution` 改成 768 .___**

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export OUTPUT_DIR="sd-pokemon-model-lora"

python train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=visualdl \
  --checkpointing_steps=500 \
  --validation_prompt="Totoro" \
  --lora_rank=4 \
  --seed=1337 \
  --validation_epochs 10
```
**___Note: 当我使用 LoRA 训练模型的时候，我们需要使用更大的学习率，因此我们这里使用 *1e-4* 而不是 *1e-5*.___**

最终经过微调后的 LoRA 权重，我们已经上传到了 [junnyu/sd-model-finetuned-lora-a100](https://huggingface.co/junnyu/sd-model-finetuned-lora-a100). **___Note: [最终的权重](https://huggingface.co/junnyu/sd-model-finetuned-lora-a100/blob/main/paddle_lora_weights.pdparams) 只有 3 MB 的大小.___**


## 推理

经过训练， LoRA 权重可以直接加载到原始的 pipeline 中。

```python
from ppdiffusers import StableDiffusionPipeline
import paddle

model_path = "junnyu/sd-model-finetuned-lora-a100"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", paddle_dtype=paddle.float32)
# 注意：如果我们想从 HF Hub 加载权重，那么我们需要设置 from_hf_hub=True
pipe.unet.load_attn_procs(model_path, from_hf_hub=True)

prompt = "Totoro"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("Totoro.png")
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/218942887-a036c605-6ef4-495a-af83-39e4ce3e0055.png">
</p>

# 参考资料
- https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
- https://github.com/CompVis/stable-diffusion
- https://huggingface.co/lambdalabs/sd-pokemon-diffusers

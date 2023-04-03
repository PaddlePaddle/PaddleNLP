# DreamBooth训练代码

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)是一种新的文本生成图像(text2image)的“个性化”（可适应用户特定的图像生成需求）扩散模型。虽然 DreamBooth 是在 Imagen 的基础上做的调整，但研究人员在论文中还提到，他们的方法也适用于其他扩散模型。只需几张（通常 3~5 张）指定物体的照片和相应的类名（如“狗”）作为输入，并添加一个唯一标识符植入不同的文字描述中，DreamBooth 就能让被指定物体“完美”出现在用户想要生成的场景中。

<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/198022219-713bc91b-25cc-49b2-897f-b570635fd640.png">
</p>

## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。

```bash
pip install -U ppdiffusers visualdl -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### 1.2 Sks Dog 训练教程

为了下载`CompVis/stable-diffusion-v1-4`模型权重，我们需要阅读并签署相关的License。在这里我们默认用户已经阅读并签署了解了相关License，有关License及模型的详细介绍，请访问[CompVis/stable-diffusion-v1-4 card](https://huggingface.co/CompVis/stable-diffusion-v1-4)。

> License: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which our license is based.
<br>

#### 1.2.1 硬件要求
当我们开启`gradient_checkpointing`功能后（Tips：该功能可以在一定程度上减少显存消耗），我们可以在24GB显存的GPU上微调模型。如果想要使用更大的`batch_size`进行更快的训练，建议用户使用具有30GB+显存的显卡。

#### 1.2.2 单机单卡训练
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dream_image"
export OUTPUT_DIR="./dream_outputs"

python -u train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
```

`train_dreambooth.py`代码可传入的参数解释如下：
> 主要修改的参数
> * `--pretrained_model_name_or_path`: 所使用的 `Stable Diffusion` 模型权重名称或者本地下载的模型路径，目前支持了上表中的8种模型权重，我们可直接替换使用。
> * `--instance_data_dir`: 实例（物体）图片文件夹地址。
> * `--instance_prompt`: 带有特定实例（物体）的提示词描述文本，例如`a photo of sks dog`，其中dog代表实例（物体）。
> * `--class_data_dir`: 类别（class）图片文件夹地址，主要作为先验知识。
> * `--class_prompt`: 类别（class）提示词文本，该提示器要与实例（物体）是同一种类别，例如`a photo of dog`，主要作为先验知识。
> * `--num_class_images`: 事先需要从`class_prompt`中生成多少张图片，主要作为先验知识。
> * `--prior_loss_weight`: 先验`loss`占比权重。
> * `--sample_batch_size`: 生成`class_prompt`文本对应的图片所用的批次（batch size），注意，当GPU显卡显存较小的时候需要将这个默认值改成1。
> * `--with_prior_preservation`: 是否将生成的同类图片（先验知识）一同加入训练，当为`True`的时候，`class_prompt`、`class_data_dir`、`num_class_images`、`sample_batch_size`和`prior_loss_weight`才生效。
> * `--num_train_epochs`: 训练的轮数，默认值为`1`。
> * `--max_train_steps`: 最大的训练步数，当我们设置这个值后，它会重新计算所需的`num_train_epochs`轮数。
> * `--checkpointing_steps`: 每间隔多少步`（global step步数）`，保存模型权重。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的 step 中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的 batch_size 。
> * `--train_text_encoder`: 是否一同训练文本编码器的部分，默认为`False`。

> 可以修改的参数
> * `--height`: 输入给模型的图片`高度`，由于用户输入的并不是固定大小的图片，因此代码中会将原始大小的图片压缩成指定`高度`的图片，默认值为`None`。
> * `--width`: 输入给模型的图片`宽度`，由于用户输入的并不是固定大小的图片，因此代码中会将原始大小的图片压缩成指定`宽度`的图片，默认值为`None`。
> * `--resolution`: 输入给模型图片的`分辨率`，当`高度`或`宽度`为`None`时，我们将会使用`resolution`，默认值为`512`。
> * `--learning_rate`: 学习率。
> * `--scale_lr`: 是否根据GPU数量，梯度累积步数，以及批量数对学习率进行缩放。缩放公式：`learning_rate * gradient_accumulation_steps * train_batch_size * num_processes`。
> * `--lr_scheduler`: 要使用的学习率调度策略。默认为 `constant`。
> * `--lr_warmup_steps`: 用于从 0 到 `learning_rate` 的线性 `warmup` 的步数。
> * `--train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--center_crop`: 在调整图片宽和高之前是否将裁剪图像居中，默认值为`False`。
> * `--random_flip`: 是否对图片进行随机水平反转，默认值为`False`。
> * `--gradient_checkpointing`: 是否开启`gradient_checkpointing`功能，在一定程度上能够更显显存，但是会减慢训练速度。
> * `--output_dir`: 模型训练完所保存的路径，默认设置为`dreambooth-model`文件夹，建议用户每训练一个模型可以修改一下输出路径，防止先前已有的模型被覆盖了。
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
通过设置`--gpus`，我们可以指定 GPU 为 `0,1,2,3` 卡。。

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dream_image"
export OUTPUT_DIR="./dream_outputs"

python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
```

#### 1.2.4 预测生成图片

当训练完成后，模型将自动保存到`output_dir`目录，在上述例子中，我们的模型最终保存到了`dream_outputs`文件夹。我们可以使用`StableDiffusionPipeline`快速加载该模型。

```
├── train_dreambooth.py # 训练脚本
├── dream_outputs  # 我们指定的输出文件路径
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
model_path = "./dream_outputs"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt).images[0]
# 保存图片，我们可以查看 yoda-pokemon.png 图片。
image.save("sks-dog.png")
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/198040284-82de104d-f353-46a1-95d8-06b3d30bc31d.png">
</p>

### 给模型引入先验知识（图片）一同训练
`with_prior_preservation`这个参数主要用于防止模型过拟合以及语言出现语义理解偏差（如，原始防止模型将`狗`误理解成了其他一种`动物`）。请参阅论文以了解更多信息。对于该种训练方式，我们首先使用带有类别提示的模型生成对应的图像，然后在训练期间将这些图像与我们自己准备的数据一起使用。
根据论文，建议生成 num_epochs * num_samples 张图像，其中 200-300 适用于绝大多数情况，因此当我们不太确定的时候，可以设置成200或300。

#### 单机训练
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dream_image"
export CLASS_DIR="./dream_class_image"
export OUTPUT_DIR="./dream_outputs_with_class"

python -u train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```
#### 预测
```python
from ppdiffusers import StableDiffusionPipeline

# 我们所需加载的模型地址，这里我们输入了训练时候使用的 output_dir 地址
model_path = "./dream_outputs_with_class"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt).images[0]
# 保存图片，我们可以查看 yoda-pokemon.png 图片。
image.save("sks-dog-with-class.png")
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/198040116-57ce9e16-53df-4f53-90b9-344627fc5fd5.png">
</p>


# 使用 LoRA 和 DreamBooth 技术进行模型训练

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
export INSTANCE_DIR="path-to-instance-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --lora_rank=4 \
  --seed=0
```

**___Note: 当我使用 LoRA 训练模型的时候，我们需要使用更大的学习率，因此我们这里使用 *1e-4* 而不是 *2e-6*.___**

最终经过微调后的 LoRA 权重，我们已经上传到了 [junnyu/lora_dreambooth_dog_example](https://huggingface.co/junnyu/lora_dreambooth_dog_example). **___Note: [最终的权重](https://huggingface.co/junnyu/lora_dreambooth_dog_example/blob/main/paddle_lora_weights.pdparams) 只有 3 MB 的大小.___**

## 推理

经过训练， LoRA 权重可以直接加载到原始的 pipeline 中。

首先我们需要加载原始的 pipeline：

```python
from ppdiffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import paddle

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
```

接下来, 我们需要使用 `load_attn_procs` 方法将 `adapter layers` 添加到 UNet 模型中。
```python
pipe.unet.load_attn_procs("junnyu/lora_dreambooth_dog_example", from_hf_hub=True)
```

最终, 我们可以使用模型进行推理预测.

```python
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("demo.png")
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/218384517-b89667f4-b5c9-4ecf-afcb-8b667c5532bb.jpg">
</p>

# 参考资料
- https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
- https://github.com/CompVis/stable-diffusion

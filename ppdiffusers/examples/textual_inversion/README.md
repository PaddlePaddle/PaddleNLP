## Textual Inversion 微调代码

[Textual inversion](https://arxiv.org/abs/2208.01618) 是一种个性化定制的文本生成图像(text2image)技术。我们只需要给模型提供 3-5 张图片，就可以训练个性化的Stable Diffusion模型。
<p align="center">
    <img src="https://textual-inversion.github.io/static/images/editing/colorful_teapot.JPG">
</p>


## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。

```bash
pip install -U ppdiffusers visualdl -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### 1.2 Cat toy 训练 object 的例子

在训练开始之前，我们需要准备需要训练的 3-5 张图片，在这里我们可以从[这里](https://huggingface.co/sd-dreambooth-library/cat-toy/tree/main/concept_images) 下载到所需要的图片，然后将里面的内容保存到一个文件夹`cat_toy_images`中。
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/196325636-3ad872b2-4e84-4169-9831-8c8aa6d72a94.png" height=40% width=40%>
</p>


#### 1.2.1 硬件要求
下面的代码需要具有16GB的显卡才可以进行微调成功。

#### 1.2.2 单机单卡训练
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# 这里需要输入刚才下载的图片所保存的文件目录
export DATA_DIR="cat_toy_images"

python -u train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed 42 \
  --output_dir="textual_inversion_cat"
```

`train_textual_inversion.py`代码可传入的参数解释如下：
> 主要修改的参数
> * `--pretrained_model_name_or_path`: 所使用的Stable Diffusion模型权重名称或者本地下载的模型路径，目前支持了上表中的8种模型权重，我们可直接替换使用。
> * `--train_data_dir`: 训练数据文件夹所在的地址，上述例子我们使用了`cat_toy_image`目录。
> * `--placeholder_token`: 用来表示所需训练`物体(object)`的占位符token，上述例子我们使用了`<cat-toy>`这个占位符，建议用户以这种带有`< >`和`-`的形式设计占位符。
> * `--initializer_token`: 用来初始化占位符token的一个token。通过给`placeholder_token`指定`initializer_token`，模型可以快速掌握新`placeholder_token`所表示的意思，从而加速模型的学习速度。注：上述例子我们使用`toy`这个单词初始化了`<cat-toy>`这个token。这样，在训练的时候模型就会有一定的先验知识，知道`<cat-toy>`是一种`玩具toy`。
> * `--learnable_property`: 学习的一种方式，可以从`["object", "style"]`进行选择，其中`object`表示学习物体，我们可以让模型根据我们提供的3-5图片，学习到对应的新物体。`style`表示学习风格，同样我们可以使得模型学习到某种绘画的风格。
> * `--num_train_epochs`: 训练的轮数，默认值为`100`。
> * `--max_train_steps`: 最大的训练步数，当我们设置这个值后，它会重新计算所需的`num_train_epochs`轮数。
> * `--save_steps`: 每间隔多少步`（global step步数）`，保存学习到的文件`learned_embeds.pdparams`。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的step中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的batch_size。
> * `--enable_xformers_memory_efficient_attention`: 是否开启`xformers`，开启后训练速度会变慢，但是能够节省显存。注意我们需要安装develop版本的paddlepaddle！

> 可以修改的参数
> * `--language`: 模型的语言，`zh`、`en`或`zh_en`，当我们使用中文模型时候，请设置成`zh`。
> * `--learning_rate`: 学习率。
> * `--scale_lr`: 是否根据GPU数量，梯度累积步数，以及批量数对学习率进行缩放。缩放公式：`learning_rate * gradient_accumulation_steps * train_batch_size * num_processes`。
> * `--lr_scheduler`: 要使用的学习率调度策略。默认为 `constant`。
> * `--lr_warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
> * `--train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--center_crop`: 在调整图片宽和高之前是否将裁剪图像居中，默认值为`False`。
> * `--height`: 输入给模型的图片`高度`，由于用户输入的并不是固定大小的图片，因此代码中会将原始大小的图片压缩成指定`高度`的图片，默认值为`None`。
> * `--width`: 输入给模型的图片`宽度`，由于用户输入的并不是固定大小的图片，因此代码中会将原始大小的图片压缩成指定`宽度`的图片，默认值为`None`。
> * `--resolution`: 输入给模型图片的`分辨率`，当`高度`或`宽度`为`None`时，我们将会使用`resolution`，默认值为`512`。
> * `--repeats`: 由于图片数量只有3-5张，因此我们需要重复训练图片数据，默认设置为重复`100遍`。
> * `--gradient_checkpointing`: 是否开启`gradient_checkpointing`功能，在一定程度上能够更显显存，但是会减慢训练速度。
> * `--output_dir`: 模型训练完所保存的路径，默认设置为`text-inversion-model`文件夹，建议用户每训练一个模型可以修改一下输出路径，防止先前已有的模型被覆盖了。
> * `--validation_prompt`: 训练过程中评估所使用的prompt文本。
> * `--validation_epochs`: 每隔多少个epoch评估模型。


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
通过设置`--gpus`，我们可以指定 GPU 为 `0,1,2,3` 卡。这里我们只训练了`1000step`，因为这里的`1000 step x 4卡`近似于`单卡训练 4000 step`。

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# 这里需要输入刚才下载的图片所保存的文件目录
export DATA_DIR="cat_toy_images"

python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed 42 \
  --output_dir="textual_inversion_cat" \
  --validation_prompt "A <cat-toy> backpack" \
  --validation_epochs 1
```


#### 1.2.4 预测生成图片

（1）加载output_dir保存的模型权重
当训练完成后，模型将自动保存到`output_dir`目录，在上述例子中，我们的模型最终保存到了`textual_inversion_cat`文件夹。我们可以使用`StableDiffusionPipeline`快速加载该模型。

```python
from ppdiffusers import StableDiffusionPipeline

# 我们所需加载的模型地址，这里我们输入了训练时候使用的 output_dir 地址
model_path = "textual_inversion_cat"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

# 注意<cat-toy>这就是我们训练模型时候定义的token。
prompt = "A <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-backpack.png")
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/196341119-f1121a79-9a93-4ab9-90f6-98b1dc477b51.png" width=40% height=40%>
</p>

（2）加载已有的`learned_embeds.pdparams`权重

```python
import paddle
from ppdiffusers import StableDiffusionPipeline
# 我们所需加载的模型地址，这里我们加载了我们微调模型所使用的权重
model_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

# 需要加载的风格或物体的权重
learned_embeded_path = "./textual_inversion_cat/learned_embeds-steps-1000.pdparams"
for token, embeds in paddle.load(learned_embeded_path).items():
    pipe.tokenizer.add_tokens(token)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    with paddle.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[token_id] = embeds

print(token)
# <cat-toy>
prompt = "A <cat-toy> backpack"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-backpack.png")
```

### 1.3 huang-guang-jian 训练 style 的例子

在训练开始之前，我们需要准备需要训练的 3-5 张图片，在这里我们可以从[这里](https://huggingface.co/sd-concepts-library/huang-guang-jian
) 下载到所需要的图片，然后将里面的内容保存到一个文件夹`huang_guang_jian_images`中。
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/196342920-8ee67ce9-d8ff-41b5-844e-1c57763680a5.png" width=40% height=40%>
</p>


#### 1.3.1 单机单卡训练
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# 这里需要输入刚才下载的图片所保存的文件目录
export DATA_DIR="huang_guang_jian_images"

python -u train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<huang-guang-jian-style>" --initializer_token="style" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed 42 \
  --output_dir="huang_guang_jian_style"
```

参数解释，请参考1.2部分内容。

#### 1.3.2 单机多卡训练
通过设置`--gpus`，我们可以指定 GPU 为 `0,1,2,3` 卡。这里我们只训练了`1000step`，因为这里的`1000 step x 4卡`近似于`单卡训练 4000 step`。

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# 这里需要输入刚才下载的图片所保存的文件目录
export DATA_DIR="huang_guang_jian_images"

python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<huang-guang-jian-style>" --initializer_token="style" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed 42 \
  --output_dir="huang_guang_jian_style"
```

#### 1.3.3 预测生成图片

（1）加载output_dir保存的模型权重
当训练完成后，模型将自动保存到`output_dir`目录，在上述例子中，我们的模型最终保存到了`huang_guang_jian_style`文件夹。我们可以使用`StableDiffusionPipeline`快速加载该模型。

```python
from ppdiffusers import StableDiffusionPipeline

# 我们所需加载的模型地址，这里我们输入了训练时候使用的 output_dir 地址
model_path = "huang_guang_jian_style"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

# 注意<huang-guang-jian-style>这就是我们训练模型时候定义的token。
prompt = "A pretty girl in <huang-guang-jian-style>"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("huang-guang-jian-girl.png")
```

（2）加载已有的`learned_embeds.pdparams`权重

```python
import paddle
from ppdiffusers import StableDiffusionPipeline
# 我们所需加载的模型地址，这里我们加载了我们微调模型所使用的权重
model_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

# 需要加载的风格或物体的权重
learned_embeded_path = "./huang_guang_jian_style/learned_embeds-steps-1000.pdparams"
for token, embeds in paddle.load(learned_embeded_path).items():
    pipe.tokenizer.add_tokens(token)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    with paddle.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[token_id] = embeds

print(token)
# <huang-guang-jian-style>
prompt = "A pretty girl in <huang-guang-jian-style>"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("huang-guang-jian-girl.png")
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/50394665/196343736-2cca0efb-28c6-44fc-a0e0-44f582f805c1.png" width=40% height=40%>
</p>

## 2 参考资料
- https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion
- https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb
- https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb
- https://huggingface.co/sd-concepts-library

## Latent Diffusion Model 从零训练代码

本教程带领大家如何开启32层的**Latent Diffusion Model**的训练（支持切换`中文`和`英文`分词器）。

___注意___:
___官方32层`CompVis/ldm-text2im-large-256`的Latent Diffusion Model使用的是vae，而不是vqvae！而Huggingface团队在设计目录结构的时候把文件夹名字错误的设置成了vqvae！为了与Huggingface团队保持一致，我们同样使用了vqvae文件夹命名！___

## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。
```bash
# paddlepaddle-gpu>=2.4.1
python -m pip install paddlepaddle-gpu==2.4.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
```

### 1.2 准备数据

#### laion400m_en.filelist文件内部格式如下所示
自己准备好处理后的数据，并且将文件放置于`/data/laion400m/`目录，其中里面的每个part的前三列为`caption文本描述, 占位符空, base64编码的图片`，`caption, _, img_b64 = vec[:3]`。

注意，当前`laion400m_en.filelist`只存放了10条数据路径，如果想要更多数据的话，请运行`python write_filelist.py`代码，运行后会生成6万条数据路径。
```
/data/laion400m/part-00000.gz
/data/laion400m/part-00001.gz
/data/laion400m/part-00002.gz
/data/laion400m/part-00003.gz
/data/laion400m/part-00004.gz
/data/laion400m/part-00005.gz
/data/laion400m/part-00006.gz
/data/laion400m/part-00007.gz
/data/laion400m/part-00008.gz
/data/laion400m/part-00009.gz
```
#### train.filelist.list训练文件内部格式如下所示
我们提供了`laion400m_en.filelist`，当然也可以存放其他`filelist`
```
./data/filelist/laion400m_en.filelist
```
Tips: 我们可以选择下载demo数据
- 删除当前目录下的`data`;
- 下载demo数据`wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz`；
- 解压demo数据`tar -zxvf laion400m_demo_data.tar.gz`

### 1.3 使用trainner开启训练
#### 1.3.1 硬件要求
Tips：
- FP32 在 40GB 的显卡上可正常训练。

#### 1.3.2 单机单卡训练
```bash
python -u train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 1000000000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 50 \
    --seed 23 \
    --dataloader_num_workers 6 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1
```


`train_txt2img_laion400m_trainer.py`代码可传入的参数解释如下：
> * `--vae_name_or_path`: 预训练`vae`模型名称或地址，`CompVis/stable-diffusion-v1-4/vae`为`kl-8.ckpt`，程序将自动从BOS上下载预训练好的权重。
> * `--text_encoder_config_file`: `ldmbert`的config配置文件地址，默认为`./config/ldmbert.json`。
> * `--unet_config_file`: `unet`的config配置文件地址，默认为`./config/unet.json`。
> * `--pretrained_model_name_or_path`: 加载预训练模型的名称或本地路径，如`CompVis/ldm-text2im-large-256`，`pretrained_model_name_or_path`的优先级高于`vae_name_or_path`, `text_encoder_config_file`和`unet_config_file`。
> * `--per_device_train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的step中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的batch_size。
> * `--learning_rate`: 学习率。
> * `--weight_decay`: AdamW优化器的`weight_decay`。
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
> * `--file_list`: file_list文件地址。
> * `--num_inference_steps`: 推理预测时候使用的步数。
> * `--model_max_length`: `tokenizer`中的`model_max_length`参数，超过该长度将会被截断。
> * `--tokenizer_name`: 我们需要使用的`tokenizer_name`，我们可以使用英文的分词器`bert-base-uncased`，也可以使用中文的分词器`ernie-1.0`。
> * `--use_ema`: 是否对`unet`使用`ema`，默认为`False`。
> * `--max_grad_norm`: 梯度剪裁的最大norm值，`-1`表示不使用梯度裁剪策略。
> * `--recompute`: 是否开启重计算，(`bool`, 可选, 默认为 `False`)，在开启后我们可以增大batch_size，注意在小batch_size的条件下，开启recompute后显存变化不明显，只有当开大batch_size后才能明显感受到区别。
> * `--fp16`: 是否使用 fp16 混合精度训练而不是 fp32 训练。(`bool`, 可选, 默认为 `False`)
> * `--fp16_opt_level`: 混合精度训练模式，可为``O1``或``O2``模式，默认``O1``模式，默认O1. 只在fp16选项开启时候生效。
> * `--enable_xformers_memory_efficient_attention`: 是否开启`xformers`，开启后训练速度会变慢，但是能够节省显存。注意我们需要安装develop版本的paddlepaddle！


#### 1.3.3 单机多卡训练 (多机多卡训练，仅需在 paddle.distributed.launch 后加个 --ips IP1,IP2,IP3,IP4)
```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 1000000000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 50 \
    --seed 23 \
    --dataloader_num_workers 6 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1
```

### 1.4 自定义训练逻辑开启训练
#### 1.4.1 单机单卡训练
```bash
python -u train_txt2img_laion400m_no_trainer.py \
    --output_dir ./laion400m_pretrain_output_no_trainer \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 1000000000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 50 \
    --save_steps 5000 \
    --seed 23 \
    --dataloader_num_workers 6 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1
```

`train_txt2img_laion400m_no_trainer.py`代码可传入的参数解释如下：
> 主要修改的参数
> * `--vae_name_or_path`: 预训练`vae`模型名称或地址，`CompVis/stable-diffusion-v1-4/vae`为`kl-8.ckpt`，程序将自动从BOS上下载预训练好的权重。
> * `--text_encoder_config_file`: `ldmbert`的config配置文件地址，默认为`./config/ldmbert.json`。
> * `--unet_config_file`: `unet`的config配置文件地址，默认为`./config/unet.json`。
> * `--pretrained_model_name_or_path`: 加载预训练模型的名称或本地路径，如`CompVis/ldm-text2im-large-256`，`pretrained_model_name_or_path`的优先级高于`vae_name_or_path`, `text_encoder_config_file`和`unet_config_file`。
> * `--per_device_train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的step中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的batch_size。
> * `--learning_rate`: 学习率。
> * `--weight_decay`: AdamW优化器的`weight_decay`。
> * `--max_steps`: 最大的训练步数。
> * `--save_steps`: 每间隔多少步`（global step步数）`，保存模型。
> * `--lr_scheduler_type`: 要使用的学习率调度策略。默认为 `constant`。
> * `--warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
> * `--image_logging_steps`: 每隔多少步，log训练过程中的图片，默认为`1000`步，注意`image_logging_steps`需要是`logging_steps`的整数倍。
> * `--logging_steps`: logging日志的步数，默认为`50`步。
> * `--output_dir`: 模型保存路径。
> * `--seed`: 随机种子，为了可以复现训练结果，Tips：当前paddle设置该随机种子后仍无法完美复现。
> * `--dataloader_num_workers`: Dataloader所使用的`num_workers`参数。
> * `--file_list`: file_list文件地址。
> * `--num_inference_steps`: 推理预测时候使用的步数。
> * `--model_max_length`: `tokenizer`中的`model_max_length`参数，超过该长度将会被截断。
> * `--tokenizer_name`: 我们需要使用的`tokenizer_name`。
> * `--use_ema`: 是否对`unet`使用`ema`，默认为`False`。
> * `--max_grad_norm`: 梯度剪裁的最大norm值，`-1`表示不使用梯度裁剪策略。
> * `--recompute`: 是否开启重计算，(`bool`, 可选, 默认为 `False`)，在开启后我们可以增大batch_size，注意在小batch_size的条件下，开启recompute后显存变化不明显，只有当开大batch_size后才能明显感受到区别。
> * `--fp16`: 是否使用 fp16 混合精度训练而不是 fp32 训练。(`bool`, 可选, 默认为 `False`)
> * `--fp16_opt_level`: 混合精度训练模式，可为``O1``或``O2``模式，默认``O1``模式，默认O1. 只在fp16选项开启时候生效。
> * `--enable_xformers_memory_efficient_attention`: 是否开启`xformers`，开启后训练速度会变慢，但是能够节省显存。注意我们需要安装develop版本的paddlepaddle！

#### 1.4.2 单机多卡训练
```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_laion400m_no_trainer.py \
    --output_dir ./laion400m_pretrain_output_no_trainer \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 1000000000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 50 \
    --save_steps 5000 \
    --seed 23 \
    --dataloader_num_workers 6 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1
```

## 2 模型推理

待模型训练完毕，会在`output_dir`保存训练好的模型权重，我们可以使用`generate_pipelines.py`生成推理所使用的`Pipeline`。
```bash
python generate_pipelines.py \
    --model_file ./laion400m_pretrain_output_no_trainer/model_state.pdparams \
    --output_path ./ldm_pipelines \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file ./config/ldmbert.json \
    --unet_config_file ./config/unet.json \
    --tokenizer_name_or_path bert-base-uncased \
    --model_max_length 77
```
`generate_pipelines.py`代码可传入的参数解释如下：
> * `--model_file`: 我们使用`train_txt2img_laion400m_trainer.py`代码，训练好所得到的`model_state.pdparams`文件。
> * `--output_path`: 生成的pipeline所要保存的路径。
> * `--vae_name_or_path`: 使用的`vae`的名字或者本地路径，注意我们需要里面的`config.json`文件。
> * `--text_encoder_config_file`: 文本编码器的`config`配置文件。
> * `--unet_config_file`: `unet`的`config`配置文件。
> * `--tokenizer_name_or_path`: 所使用的`tokenizer`名称或者本地路径，名称可以是`bert-base-uncased`, `bert-base-chinese`, `ernie-1.0`等。
> * `--model_max_length`: `tokenizer`中的`model_max_length`参数，超过该长度将会被截断。


输出的模型目录结构如下：
```shell
├── ldm_pipelines  # 我们指定的输出文件路径
    ├── model_index.json # 模型index文件
    ├── vqvae # vae权重文件夹！实际是vae模型，文件夹名字与HF保持了一致！
        ├── model_state.pdparams
        ├── config.json
    ├── bert # ldmbert权重文件夹
        ├── model_config.json
        ├── model_state.pdparams
    ├── unet # unet权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── scheduler # ddim scheduler文件夹
        ├── scheduler_config.json
    ├── tokenizer # bert tokenizer文件夹
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        ├── vocab.txt
```

在生成`Pipeline`的权重后，我们可以使用如下的代码进行推理。

```python
from ppdiffusers import LDMTextToImagePipeline
model_name_or_path = "./ldm_pipelines"
pipe = LDMTextToImagePipeline.from_pretrained(model_name_or_path)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, guidance_scale=7.5).images[0]
image.save("astronaut_rides_horse.png")
```

当然，我们也可以使用训练好的模型在`coco en 1k`数据集上生成图片。
首先我们需要下载`mscoco.en.1k`文件。
```bash
wget https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mscoco.en.1k
```
然后可以`generate_images.py`文件生成对应的图片。
```bash
python generate_images.py \
    --model_name_or_path ./ldm_pipelines \
    --file ./mscoco.en.1k \
    --batch_size 16 \
    --save_path ./outputs \
    --guidance_scales 3 4 5 6 7 8 \
    --seed 42 \
    --scheduler_type ddim \
    --height 256 \
    --width 256 \
    --num_inference_steps 50 \
    --device gpu
```
`generate_images.py`代码可传入的参数解释如下：
> * `--model_name_or_path`: 我们需要评估的模型名称或地址，这里我们使用上一步生成的`ldm_pipelines`。
> * `--file`: 需要评估的文件，我们可以从[这里](https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mscoco.en.1k)下载。
> * `--batch_size`: 生成图片所使用的batch_size。
> * `--save_path`: 生成的图片所要保存的路径。
> * `--guidance_scales`: guidance_scales值，我们可以输入3 4 5 6 7 8。
> * `--seed`: 为了保证不同guidance_scales值，能够使用相同的`latents`初始值, `-1`表示不使用随机种子。
> * `--scheduler_type`: 采样器的类型，支持`ddim`, `pndm`, `euler-ancest` 和 `lms`。
> * `--num_inference_steps`: 推理预测时候使用的步数。
> * `--height`: 生成图片的高度。
> * `--width`: 生成图片的宽度。
> * `--device`: 使用的设备，可以是`gpu`, `cpu`, `gpu:0`, `gpu:1`等。


输出的图片目录如下：
```shell
├── outputs  # 我们指定的输出文件路径
    ├── mscoco.en_g3 # guidance_scales为3的输出图片
        ├── 00000_000.png
        ├── 00001_000.png
        ......
        ├── 00999_000.png
    ├── mscoco.en_g4 # guidance_scales为4的输出图片
        ├── 00000_000.png
        ├── 00001_000.png
        ......
        ├── 00999_000.png
    ......
    ├── mscoco.en_g8 # guidance_scales为8的输出图片
        ├── 00000_000.png
        ├── 00001_000.png
        ......
        ├── 00999_000.png
```

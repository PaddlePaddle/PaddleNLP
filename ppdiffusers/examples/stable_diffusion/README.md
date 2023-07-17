## Stable Diffusion Model 从零训练代码

本教程带领大家如何从零预训练Stable Diffusion模型。

## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。
```bash
# paddlepaddle-gpu>=2.5.0rc1
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
- FP32 和 BF16 在 40GB 的显卡上可正常训练。

#### 1.3.2 单机单卡训练
```bash
export FLAGS_conv_workspace_size_limit=4096
# 是否开启ema
export FLAG_USE_EMA=0
# 是否开启recompute
export FLAG_RECOMPUTE=1
# 是否开启xformers
export FLAG_XFORMERS=1
python -u train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_steps 200000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 10 \
    --resolution 256 \
    --save_steps 10000 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 8 \
    --pretrained_model_name_or_path ./CompVis-stable-diffusion-v1-4-paddle-init \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --disable_tqdm True \
    --bf16 True
```


`train_txt2img_laion400m_trainer.py`代码可传入的参数解释如下：
> * `--vae_name_or_path`: 预训练`vae`模型名称或地址，`CompVis/stable-diffusion-v1-4/vae`为`kl-8.ckpt`，程序将自动从BOS上下载预训练好的权重，默认值为`None`。
> * `--text_encoder_name_or_path`: 预训练`text_encoder`模型名称或地址，当前仅支持`CLIPTextModel`，默认值为`None`。
> * `--unet_name_or_path`: 预训练`unet`模型名称或地址，默认值为`None`。
> * `--pretrained_model_name_or_path`: 加载预训练模型的名称或本地路径，如`CompVis/stable-diffusion-v1-4`，`vae_name_or_path`, `text_encoder_name_or_path`和`unet_name_or_path`的优先级高于`pretrained_model_name_or_path`。
> * `--per_device_train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的step中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的batch_size。
> * `--learning_rate`: 学习率。
> * `--unet_learning_rate`: `unet`的学习率，这里的学习率优先级将会高于`learning_rate`，默认值为`None`。
> * `--train_text_encoder`: 是否同时训练`text_encoder`，默认值为`False`。
> * `--text_encoder_learning_rate`: `text_encoder`的学习率，默认值为`None`。
> * `--weight_decay`: AdamW优化器的`weight_decay`。
> * `--max_steps`: 最大的训练步数。
> * `--save_steps`: 每间隔多少步`（global step步数）`，保存模型。
> * `--save_total_limit`: 最多保存多少个模型。
> * `--lr_scheduler_type`: 要使用的学习率调度策略。默认为 `constant`。
> * `--warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
> * `--resolution`: 预训练阶段将训练的图像的分辨率，默认为`512`。
> * `--noise_offset`: 预训练阶段生成操作时的偏移量，默认为`0`。
> * `--snr_gamma`: 平衡损失时使用的SNR加权gamma值。建议为`5.0`, 默认为`None`。更多细节在这里：`https://arxiv.org/abs/2303.09556`。
> * `--input_perturbation`: 输入扰动的尺度，推荐为`0.1`，默认值为`0`。
> * `--image_logging_steps`: 每隔多少步，log训练过程中的图片，默认为`1000`步，注意`image_logging_steps`需要是`logging_steps`的整数倍。
> * `--logging_steps`: logging日志的步数，默认为`50`步。
> * `--output_dir`: 模型保存路径。
> * `--seed`: 随机种子，为了可以复现训练结果，Tips：当前paddle设置该随机种子后仍无法完美复现。
> * `--dataloader_num_workers`: Dataloader所使用的`num_workers`参数。
> * `--file_list`: file_list文件地址。
> * `--num_inference_steps`: 推理预测时候使用的步数。
> * `--model_max_length`: `tokenizer`中的`model_max_length`参数，超过该长度将会被截断。
> * `--tokenizer_name`: 我们需要使用的`tokenizer_name`。
> * `--prediction_type`: 预测类型，可从`["epsilon", "v_prediction"]`选择。
> * `--use_ema`: 是否对`unet`使用`ema`，默认为`False`。
> * `--max_grad_norm`: 梯度剪裁的最大norm值，`-1`表示不使用梯度裁剪策略。
> * `--recompute`: 是否开启重计算，(`bool`, 可选, 默认为 `False`)，在开启后我们可以增大batch_size，注意在小batch_size的条件下，开启recompute后显存变化不明显，只有当开大batch_size后才能明显感受到区别。
> * `--bf16`: 是否使用 bf16 混合精度训练而不是 fp32 训练。(`bool`, 可选, 默认为 `False`)
> * `--fp16`: 是否使用 fp16 混合精度训练而不是 fp32 训练。(`bool`, 可选, 默认为 `False`)
> * `--fp16_opt_level`: 混合精度训练模式，可为``O1``或``O2``模式，默认``O1``模式，默认O1. 只在fp16选项开启时候生效。
> * `--enable_xformers_memory_efficient_attention`: 是否开启`xformers`，开启后训练速度会变慢，但是能够节省显存。注意我们需要安装develop版本的paddlepaddle！
> * `--only_save_updated_model`: 是否仅保存经过训练的权重，比如保存`unet`、`ema版unet`、`text_encoder`，默认值为`True`。


#### 1.3.3 单机多卡训练 (多机多卡训练，仅需在 paddle.distributed.launch 后加个 --ips IP1,IP2,IP3,IP4)
```bash
export FLAGS_conv_workspace_size_limit=4096
# 是否开启ema
export FLAG_USE_EMA=0
# 是否开启recompute
export FLAG_RECOMPUTE=1
# 是否开启xformers
export FLAG_XFORMERS=1
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_steps 200000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 10 \
    --resolution 256 \
    --save_steps 10000 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 8 \
    --pretrained_model_name_or_path ./CompVis-stable-diffusion-v1-4-paddle-init \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --disable_tqdm True \
    --bf16 True
```



## 2 模型推理
待模型训练完毕，会在`output_dir`保存训练好的模型权重.

```python
from ppdiffusers import StableDiffusionPipeline, UNet2DConditionModel
unet_model_name_or_path = "./output/checkpoint-5000/unet"
unet = UNet2DConditionModel.from_pretrained(unet_model_name_or_path
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None, unet=unet)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, guidance_scale=7.5, width=256, height=256).images[0]
image.save("astronaut_rides_horse.png")
```

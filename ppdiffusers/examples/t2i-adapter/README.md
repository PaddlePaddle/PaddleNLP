# T2I-Adapter
[T2I-Adapter](https://arxiv.org/abs/2302.08453) 是一种通过添加额外条件来控制扩散模型的神经网络结构。它通过将T2I（Text2Image）模型中的内部知识与外部控制信号对齐，根据不同条件训练各种适配器（Adapter），从而实现丰富的控制和编辑效果。
<p align="center">
    <img src="https://github.com/TencentARC/T2I-Adapter/blob/main/assets/overview1.png?raw=true">
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

# 训练与推理
## Adapter模型训练
下面我们将以pose2canny任务为例，介绍如何训练相应的Adapter模型。
### 数据准备
请自行按照`adapter/data_preprocess.py`的数据处理逻辑准备好数据，并且将文件放置于`/data`目录，数据中需包含原图像、控制文本、控制图像等信息。

Tips: 我们可以选择下载demo数据并替换掉`/data`目录
- 下载demo数据`wget https://paddlenlp.bj.bcebos.com/models/community/westfish/t2i-adapter/t2i-adapter-data-demo.zip`；

### 单机单卡训练
```bash
export FLAGS_conv_workspace_size_limit=4096
python -u -m train_t2i_adapter_trainer.py \
    --do_train \
    --output_dir ./sd15_openpose \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --max_steps 50000 \
    --logging_steps 1 \
    --image_logging_steps 500 \
    --save_steps 50 \
    --save_total_limit 1000 \
    --seed 4096 \
    --dataloader_num_workers 0 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_list ./data/train.openpose.filelist \
    --recompute False --use_ema False \
    --control_type raw \
    --data_format img2img \
    --use_paddle_conv_init False \
    --overwrite_output_dir
```
`train_t2i_adapter_trainer.py`关键传入的参数解释如下：
> * `--pretrained_model_name_or_path`: 加载预训练模型的名称或本地路径，如`runwayml/stable-diffusion-v1-5`，`pretrained_model_name_or_path`的优先级高于`vae_name_or_path`, `text_encoder_name_or_path`和`unet_name_or_path`。
> * `--per_device_train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，以期在梯度累积的step中减少多卡之间梯度的通信量，减少更新的次数，扩大训练的batch_size。
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
> * `--use_paddle_conv_init`: 是否使用`paddle`的卷积初始化策略，默认值为 `False`，否则将采用`Uniform`卷积初始化策略。
> * `--recompute`: 是否开启重计算，(`bool`, 可选, 默认为 `False`)，在开启后我们可以增大`batch_size`。
> * `--fp16`: 是否使用 fp16 混合精度训练而不是 fp32 训练。(`bool`, 可选, 默认为 `False`)
> * `--fp16_opt_level`: 混合精度训练模式，可为``O1``或``O2``模式，默认``O1``模式，默认O1. 只在fp16选项开启时候生效。
> * `--is_ldmbert`: 是否使用`ldmbert`作为`text_encoder`，默认为`False`，即使用 `clip text_encoder`。
> * `--overwrite_output_dir`: 加入该参数之后，将覆盖之前的模型保存路径，不会自动恢复训练。

### 单机多卡训练 (多机多卡训练，仅需在 paddle.distributed.launch 后加个 --ips IP1,IP2,IP3,IP4)
```bash
export FLAGS_conv_workspace_size_limit=4096
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_t2i_adapter_trainer.py \
    --do_train \
    --output_dir ./sd15_openpose \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --max_steps 50000 \
    --logging_steps 1 \
    --image_logging_steps 500 \
    --save_steps 50 \
    --save_total_limit 1000 \
    --seed 4096 \
    --dataloader_num_workers 0 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_list ./data/train.openpose.filelist \
    --recompute False --use_ema False \
    --control_type raw \
    --data_format img2img \
    --use_paddle_conv_init False \
    --overwrite_output_dir
```

## 模型推理
### 简易推理
待模型训练完毕，会在`output_dir`保存训练好的模型权重，我们可以使用如下的代码进行推理。
```python
from ppdiffusers import StableDiffusionAdapterPipeline, Adapter
from ppdiffusers.utils import load_image
adapter = Adapter.from_pretrained("./sd15_control/checkpoint-12000/adapter")
pipe = StableDiffusionAdapterPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", adapter = adapter, safety_checker=None)
pose_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/westfish/t2i-adapter/test/man-openpose.png")
img = pipe(prompt="a beautiful girl", image=pose_image, guidance_scale=9, num_inference_steps=50).images[0]
img.save("demo.png")
```

### 测试集推理
我们可以使用如下命令针对相应的测试集（需符合`adapter/data_preprocess.py`的数据处理逻辑）进行测试。
```
python generate.py \
    --adapter_model_name_or_path westfish/sd-v1-4-adapter-openpose \
    --sd_model_name_or_path lllyasviel/sd-controlnet-openpose \
    --save_path your/output/path \
    --num_inference_steps 50 \
    --scheduler_type ddim \
    --height=512 \
    --width=512 \
    --device gpu \
    --max_generation_limits 1000 \
    --use_text_cond True \
    --generate_control_image_processor_type openpose \
    --file data/test.openpose.filelist \
    --generate_data_format img2img \
```
`generate.py`关键传入的参数解释如下：
> * `--use_controlnet`: 是否采用ControlNet来进行条件控制，默认为`False`，即默认使用Adapter来进行条件控制。
> * `--adapter_model_name_or_path`: Adapter采用的的的模型名称或地址。
> * `--sd_model_name_or_path`: Stable Diffusion采用的的的模型名称或地址。
> * `--file`: 需要测试的数据。
> * `--batch_size`: 生成图片所使用的batch_size。
> * `--save_path`: 生成的图片所要保存的路径。
> * `--guidance_scales`: guidance_scales值，默认为[3 5 7]。
> * `--num_inference_steps`: 推理预测时候使用的步数。
> * `--scheduler_type`: 采样器的类型，支持`ddim`, `pndm`, `euler-ancest` 和 `lms`。
> * `--height`: 生成图片的高，默认为512。
> * `--width`: 生成图片的宽，默认为512。
> * `--seed`: 随机种子。
> * `--device`: 使用的设备，可以是`gpu`, `cpu`, `gpu:0`, `gpu:1`等。
> * `--max_generation_limits`: 每次最多生成的个数。
> * `--use_text_cond`: 是否使用数据集中自带的文本提示词，默认为`True`。
> * `--use_default_neg_text_cond`: 是否使用默认的负提示词，默认为`True`。
> * `--generate_control_image_processor_type`: 控制生成的类型，可选择`canny`、`openpose`。
> * `--generate_data_format`: 数据控制类型，当`generate_control_image_processor_type`为`canny`是设置为`default`，其他情况下设置为`img2img`。

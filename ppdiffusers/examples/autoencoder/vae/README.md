## AutoEncoderKL(VAE) 从零训练代码

本教程带领大家如何开启[f8encoder_f16decoder](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/examples/autoencoder/vae/config/f8encoder_f16decoder.yaml)架构的AutoEncoderKL (VAE) 模型。


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
自己准备好处理后的数据，并且将文件放置于`/data/laion400m/`目录，其中里面的每个part的前三列为`占位符空, caption文本描述, 占位符空, base64编码的图片`，`_, caption, _, img_b64 = vec[:4]`。

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

### 1.3 Encoder热启，Decoder从零开启训练
Tips：
- FP32 在 40GB 的显卡上可正常训练，在下面的配置条件下，显存占用约 29G。

#### 1.3.1 单机单卡训练
```bash
python -u train_vae.py \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --ignore_keys decoder. \
    --vae_config_file config/vae.json \
    --freeze_encoder \
    --input_size 256 256 \
    --max_train_steps 100000000000 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --num_workers 8 \
    --logging_steps 100 \
    --save_steps 2000 \
    --image_logging_steps 500 \
    --disc_start 50001 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512
```

`train_vae.py`代码可传入的参数解释如下：
> * `--pretrained_model_name_or_path`: 加载预训练模型的名称或本地路径，当我们设置成`CompVis/stable-diffusion-v1-4`后，我们会加载自动加载此模型VAE（kl-8.ckpt）部分的预训练权重。例如：在上面的训练代码中，我们（1）加载了 `kl-8.ckpt` 的 `encoder` 部分权重 （设置 `pretrained_model_name_or_path` 参数），（2）修改了模型 `decoder` 部分的结构 （指定了 `vae_config_file`），（3）删除了不希望加载的预训练权重（设置`ignore_keys`，会自动删除前缀为`ignore_keys`的模型参数）。
> * `--from_scratch`: 是否所有权重均从零初始化开启训练。
> * `--scale_lr`: 是否对学习率进行缩放，缩放公式：`ngpus*batch_size*learning_rate`。
> * `--batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--input_size`: `encoder`处接收图片的`height`和`width`，我们在训练不对等层数的`encoder-decoder`结构的`VAE`模型时候才会指定这个参数。
> * `--learning_rate`: 学习率。
> * `--max_train_steps`: 最大的训练步数。
> * `--save_steps`: 每间隔多少步`（global step步数）`，保存模型。
> * `--image_logging_steps`: 每隔多少步，log训练过程中的图片，默认为`500`步，注意`image_logging_steps`需要是`logging_steps`的整数倍。
> * `--logging_steps`: logging日志的步数，默认为`100`步，注意，这里log的日志只是单卡、单步的loss信息。
> * `--output_dir`: 模型保存路径。
> * `--seed`: 随机种子，为了可以复现训练结果，Tips：当前paddle设置该随机种子后仍无法完美复现。
> * `--num_workers`: Dataloader所使用的`num_workers`参数。
> * `--dataset_type`: 数据集类型，当前我们支持`imagenet` 和 `text_image_pair` 两种数据集，默认是`text_image_pair`。
> * `--file_list`: file_list文件地址，当我们数据集类型是`text_image_pair`时才需要指定。
> * `--disc_start`: 判别器开启训练的步数。
> * `--disc_weight`: 判别器loss的权重比例。
> * `--kl_weight`: kl_loss的权重比例。
> * `--resolution`: 训练时，图像的分辨率。
> * `--init_from_ckpt`: 是否加载预训练的ckpt权重，注意：如果我们为了严格同步pytorch的参数初始化，我们可以首先进行转换，然后再设置`init_from_ckpt`这个参数，从而加载预训练权重，如：`scripts/ldm_vae_init0_paddle/model_state.pdparams`。


#### 1.3.2 单机多卡训练 (多机多卡训练，仅需在 paddle.distributed.launch 后加个 --ips IP1,IP2,IP3,IP4)
```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_vae.py \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --ignore_keys decoder. \
    --vae_config_file config/vae.json \
    --freeze_encoder \
    --input_size 256 256 \
    --max_train_steps 100000000000 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --num_workers 8 \
    --logging_steps 100 \
    --save_steps 2000 \
    --image_logging_steps 500 \
    --disc_start 50001 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512
```

### 1.4 Encoder和Decoder从零开启训练
Tips：
- FP32 在 40GB 的显卡上可正常训练，在下面的配置条件下，显存占用约 29G。

#### 1.4.1 单机单卡训练

```bash
python -u train_vae.py \
    --from_scratch \
    --vae_config_file config/vae.json \
    --input_size 256 256 \
    --max_train_steps 100000000000 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --num_workers 8 \
    --logging_steps 100 \
    --save_steps 2000 \
    --image_logging_steps 500 \
    --disc_start 50001 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512
```
`train_vae.py`代码可传入的参数解释可参考 **1.3.1** 小节。

注意：当我们指定开启`from_scratch`后必须指定`vae_config_file`！


#### 1.4.2 单机多卡训练 (多机多卡训练，仅需在 paddle.distributed.launch 后加个 --ips IP1,IP2,IP3,IP4)

```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_vae.py \
    --from_scratch \
    --vae_config_file config/vae.json \
    --input_size 256 256 \
    --max_train_steps 100000000000 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --num_workers 8 \
    --logging_steps 100 \
    --save_steps 2000 \
    --image_logging_steps 500 \
    --disc_start 50001 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512
```

## 2 模型推理
```python
import paddle
from IPython.display import display
from ppdiffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline
from ppdiffusers.utils import load_image
from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import preprocess

def decode_image(image):
    image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]).cast('float32').numpy()
    image = StableDiffusionImg2ImgPipeline.numpy_to_pil(image)
    return image

# 我们只需要修改这里的参数配置就可以！
model_name_or_path = "./autoencoder_outputs/checkpoint-200000"
vae = AutoencoderKL.from_pretrained(model_name_or_path)
image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/vermeer.jpg")
sample_32 = preprocess(image.resize((256, 256)))
sample_64 = preprocess(image.resize((512, 512)))

with paddle.no_grad():
    # sample_32 256 x 256
    dec_32 = vae(sample_32, sample_posterior=True)[0] # must set sample_posterior = True
    img_32 = decode_image(dec_32)[0]
    display(img_32)
    # img_32 512 x 512
    img_32.save('32.jpg')

with paddle.no_grad():
    # sample_32 512 x 512
    dec_64 = vae(sample_64, sample_posterior=True)[0] # must set sample_posterior = True
    img_64 = decode_image(dec_64)[0]
    display(img_64)
    # img_64 1024 x 1024
    img_64.save('64.jpg')
```

<div align="center">
<img width="200" alt="image" src="https://user-images.githubusercontent.com/50394665/208030125-6d617506-89a0-4251-ac98-02303b35fccd.jpg">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/50394665/208030119-c1e981d6-7364-4c7c-9163-b5b810c5b224.jpg">
</div>

## 3 其他
### 3.1 ImageNet 数据集准备
The code will try to download (through [Academic
Torrents](http://academictorrents.com/)) and prepare ImageNet the first time it
is used. However, since ImageNet is quite large, this requires a lot of disk
space and time. If you already have ImageNet on your disk, you can speed things
up by putting the data into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` (which defaults to
`~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`), where `{split}` is one
of `train`/`validation`. It should have the following structure:

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── ...
├── ...
```

If you haven't extracted the data, you can also place
`ILSVRC2012_img_train.tar`/`ILSVRC2012_img_val.tar` (or symlinks to them) into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/` /
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_validation/`, which will then be
extracted into above structure without downloading it again.  Note that this
will only happen if neither a folder
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` nor a file
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/.ready` exist. Remove them
if you want to force running the dataset preparation again.

## 4 参考资料
- https://github.com/CompVis/latent-diffusion
- https://github.com/huggingface/diffusers

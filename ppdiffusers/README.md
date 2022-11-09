# PPDiffusers: Diffusers toolbox implemented based on PaddlePaddle


**PPDiffusers**是一款支持**跨模态**（如图像与语音）训练和推理的**扩散模型**（Diffusion Model）工具箱，我们借鉴了🤗 Huggingface团队的[**Diffusers**](https://github.com/huggingface/diffusers)的优秀设计，并且依托[**PaddlePaddle**](https://www.paddlepaddle.org.cn/)框架和[**PaddleNLP**](https://github.com/PaddlePaddle/PaddleNLP)自然语言处理库，打造了一款国产化的工具箱。

## 1. News 📢

* 🔥 **2022.11.04 支持 IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 和 IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1 中文权重**
* 🔥 **2022.10.27 发布 PPDiffusers仓库**


## 2. 安装
**使用 `pip` 安装**

```bash
pip install --upgrade ppdiffusers
```

**手动安装**
```bash
# 克隆paddlenlp仓库
git clone https://github.com/PaddlePaddle/PaddleNLP
# 注意：如果clone仓库非常慢的话，可以考虑使用镜像版本
# git clone https://gitee.com/paddlepaddle/PaddleNLP
# 切换目录，进入ppdiffusers文件夹
cd PaddleNLP/ppdiffusers
# 安装ppdiffusers
python setup.py install
```

## 3. 快速开始

为了快速上手使用该项目, 我们可以先阅读🤗 Huggingface团队提供的**两个notebooks** (注意国内可能无法正常打开):

- The [Getting started with Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) notebook, which showcases an end-to-end example of usage for diffusion models, schedulers and pipelines.
  Take a look at this notebook to learn how to use the pipeline abstraction, which takes care of everything (model, scheduler, noise handling) for you, and also to understand each independent building block in the library.
- The [Training a diffusers model](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) notebook summarizes diffusion models training methods. This notebook takes a step-by-step approach to training your
  diffusion models on an image dataset, with explanatory graphics.


## 4. 使用PPDiffusers快速体验Stable Diffusion模型!

Stable Diffusion 是一个**文本到图像(text-to-image)**的**潜在扩散模型(latent diffusion model, ldm)**, 该模型是由来自[CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/), [LAION](https://laion.ai/) 的工程师以及 [RunwayML](https://runwayml.com/)一起开发而完成的。该模型使用了大小为**512x512**的[LAION-5B](https://laion.ai/blog/laion-5b/)数据集子集进行训练。该模型使用了Openai开源的**CLIP ViT-L/14** 文本编码器(text_encoder)来编码提示(prompt)文本，从而作为引导条件（注意该部分权重不进行训练）。该模型使用了Unet模型（860M参数）和text encoder（123M参数），并且可以在具有4GB显存的GPU上进行推理预测。

___注意___:
___为了方便国内用户下载使用及快速体验Stable Diffusion模型，我们在百度云(BOS)上提供了paddle版本的镜像权重。注意：为了使用该模型与权重，你必须接受该模型所要求的**License**，请访问huggingface的[model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), 仔细阅读里面的**License**，然后签署该协议。___

___Tips___:
___Stable Diffusion是基于以下的License:
The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___


### 4.1 使用Stable Diffusion进行文本-图像的生成
```python
import paddle
from ppdiffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/50394665/197779466-04543823-8b83-41d6-94e8-146a7dac00d7.png">

### 4.2 使用Stable Diffusion进行由文本引导的图片-图片的生成

```python
import requests
import paddle
from PIL import Image
from io import BytesIO

from ppdiffusers import StableDiffusionImg2ImgPipeline

# 加载pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 下载初始图片
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"
# 使用fp16加快生成速度
with paddle.amp.auto_cast(True):
    image = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images[0]

image.save("fantasy_landscape.png")
```

<img width="600" alt="image" src="https://user-images.githubusercontent.com/50394665/197780044-34e6f8ca-6864-4c3d-bb99-28e0aadf867b.png">


### 4.3 使用Stable Diffusion根据文本补全图片

Tips: 下面的使用方法是旧版本的代码。
```python
import paddle
from io import BytesIO

import requests
import PIL

from ppdiffusers import StableDiffusionInpaintPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

prompt = "a cat sitting on a bench"
with paddle.amp.auto_cast(True):
    image = pipe(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75).images[0]

image.save("cat_on_bench.png")
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/50394665/197783711-ab3caf2e-5a4d-4099-8d01-d6ca80ca8e78.png">

Tips: 下面的使用方法是新版本的代码，也是官方推荐的代码，注意必须配合**runwayml/stable-diffusion-inpainting**才可正常使用。
```python
import PIL
import requests
from io import BytesIO

from ppdiffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
scheduler = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", scheduler=scheduler)

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

image.save("cat_on_bench_new.png")
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/50394665/198016801-87cec13b-0d89-41c3-aedb-c89a43d76153.png">

## 5. Credits

This library concretizes previous work by many different authors and would not have been possible without their great research and implementations. We'd like to thank, in particular, the following implementations which have helped us in our development and without which the API could not have been as polished today:
- @huggingface' diffusers library, available [here](https://github.com/huggingface/diffusers)
- @CompVis' latent diffusion models library, available [here](https://github.com/CompVis/latent-diffusion)
- @hojonathanho original DDPM implementation, available [here](https://github.com/hojonathanho/diffusion) as well as the extremely useful translation into PyTorch by @pesser, available [here](https://github.com/pesser/pytorch_diffusion)
- @ermongroup's DDIM implementation, available [here](https://github.com/ermongroup/ddim).
- @yang-song's Score-VE and Score-VP implementations, available [here](https://github.com/yang-song/score_sde_pytorch)

We also want to thank @heejkoo for the very helpful overview of papers, code and resources on diffusion models, available [here](https://github.com/heejkoo/Awesome-Diffusion-Models) as well as @crowsonkb and @rromb for useful discussions and insights.

## 6. Citation

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

## 7. License

PPDiffusers遵循[Apache-2.0开源协议](./LICENSE)。

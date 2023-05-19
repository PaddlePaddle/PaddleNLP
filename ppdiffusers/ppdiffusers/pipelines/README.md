# PPDiffusers Pipelines

Pipelines提供了一种对各种SOTA扩散模型进行各种下游任务推理的简单方式。
大多数扩散模型系统由多个独立训练的模型和高度自适应的调度器(scheduler)组成，通过pipeline我们可以很方便的对这些扩散模型系统进行端到端的推理。

举例来说， Stable Diffusion由以下组件构成:
- Autoencoder
- Conditional Unet
- CLIP text encoder
- Scheduler
- CLIPFeatureExtractor
- Safety checker

这些组件之间是独立训练或创建的，同时在Stable Diffusion的推理运行中也是必需的，我们可以通过pipelines来对整个系统进行封装，从而提供一个简洁的推理接口。

我们通过pipelines在统一的API下提供所有开源且SOTA的扩散模型系统的推理能力。具体来说，我们的pipelines能够提供以下功能：
1. 可以加载官方发布的权重，并根据相应的论文复现出与原始实现相同的输出
2. 提供一个简单的用户界面来推理运行扩散模型系统，参见[Pipelines API](#pipelines-api)部分
3. 提供易于理解的代码实现，可以与官方文档一起阅读，参见[Pipelines汇总](#Pipelines汇总)部分
4. 支持多种模态下的10+种任务，参见[任务展示](#任务展示)部分
5. 可以很容易地与社区建立联系

**【注意】** Pipelines不（也不应该）提供任何训练功能。
如果您正在寻找训练的相关示例，请查看[examples](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples).

## Pipelines汇总

下表总结了所有支持的Pipelines，以及相应的来源、任务、推理脚本。

| Pipeline                                                                                                                      | 源链接                                                                                                                       | 任务 | 推理脚本
|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|:---:|:---:|
| [alt_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/alt_diffusion)                 | [**Alt Diffusion**](https://arxiv.org/abs/2211.06679)   | *Text-to-Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-alt_diffusion.py)
| [alt_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/alt_diffusion)                 | [**Alt Diffusion**](https://arxiv.org/abs/2211.06679)   | *Image-to-Image Text-Guided Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_to_image_text_guided_generation-alt_diffusion.py)
| [audio_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/audio_diffusion)                 | [**Audio Diffusion**](https://github.com/teticio/audio-diffusion)   | *Unconditional Audio Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_audio_generation-audio_diffusion.py)
| [controlnet](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion/controlnet)                 | [**ControlNet with Stable Diffusion**](https://arxiv.org/abs/2302.05543)   | *Image-to-Image Text-Guided Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_to_image_text_guided_generation-controlnet.py)
| [dance_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/dance_diffusion)                 | [**Dance Diffusion**](https://github.com/Harmonai-org/sample-generator)                                                      | *Unconditional Audio Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_audio_generation-dance_diffusion.py)
| [ddpm](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/ddpm)                                       | [**Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2006.11239)                                             | *Unconditional Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-ddpm.py)
| [ddim](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/ddim)                                       | [**Denoising Diffusion Implicit Models**](https://arxiv.org/abs/2010.02502)                                                  | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-ddim.py)
| [latent_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/latent_diffusion)               | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)                         | *Text-to-Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-latent_diffusion.py)
| [latent_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/latent_diffusion)               | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)                         | *Super Superresolution* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/super_resolution-latent_diffusion.py)
| [latent_diffusion_uncond](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/latent_diffusion_uncond) | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)                         | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-latent_diffusion_uncond.py)
| [paint_by_example](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/paint_by_example)                                       | [**Paint by Example: Exemplar-based Image Editing with Diffusion Models**](https://arxiv.org/abs/2211.13227)                           | *Image-Guided Image Inpainting* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_guided_image_inpainting-paint_by_example.py)
| [pndm](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/pndm)                                       | [**Pseudo Numerical Methods for Diffusion Models on Manifolds**](https://arxiv.org/abs/2202.09778)                           | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-pndm.py)
| [repaint](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/repaint)                 | [**Repaint**](https://arxiv.org/abs/2201.09865)                                                      | *Image Inpainting* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_inpainting-repaint.py)
| [score_sde_ve](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/score_sde_ve)                       | [**Score-Based Generative Modeling through Stochastic Differential Equations**](https://openreview.net/forum?id=PxTIG12RRHS) | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-score_sde_ve.py)
| [semantic_stable_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/semantic_stable_diffusion)                | [**Semantic Guidance**](https://arxiv.org/abs/2301.12247)                                            | *Text-Guided Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_guided_generation-semantic_stable_diffusion.py)
| [stable_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release)                                            | *Text-to-Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-stable_diffusion.py)
| [stable_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)               | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release)                                            | *Image-to-Image Text-Guided Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_to_image_text_guided_generation-stable_diffusion.py)
| [stable_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                 | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release)                                            | *Text-Guided Image Inpainting* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_guided_image_inpainting-stable_diffusion.py)
| [stable_diffusion_2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release)                                            | *Text-to-Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-stable_diffusion_2.py)
| [stable_diffusion_2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)               | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release)                                            | *Image-to-Image Text-Guided Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_to_image_text_guided_generation-stable_diffusion_2.py)
| [stable_diffusion_2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                 | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release)                                            | *Text-Guided Image Inpainting* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_guided_image_inpainting-stable_diffusion_2.py)
| [stable_diffusion_2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                 | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release)                                            | *Text-Guided Image Upscaling* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_guided_image_upscaling-stable_diffusion_2.py)
| [stable_diffusion_2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                 | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release)                                            | *Text-Guided Image Upscaling* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_guided_image_upscaling-stable_diffusion_2.py)
| [stable_diffusion_safe](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion_safe)                 | [**Safe Stable Diffusion**](https://arxiv.org/abs/2211.05105)                                                      | *Text-to-Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-stable_diffusion_safe.py)
| [stochastic_karras_ve](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stochastic_karras_ve)       | [**Elucidating the Design Space of Diffusion-Based Generative Models**](https://arxiv.org/abs/2206.00364)                    | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-stochastic_karras_ve.py)
| [unclip](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/unclip)                 | [**UnCLIP**](https://arxiv.org/abs/2204.06125)                                                      | *Text-to-Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-unclip.py)
| [versatile_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/versatile_diffusion)                 | [**Versatile Diffusion**](https://arxiv.org/abs/2211.08332)                                                      | *Text-to-Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-versatile_diffusion.py)
| [versatile_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/versatile_diffusion)                 | [**Versatile Diffusion**](https://arxiv.org/abs/2211.08332)                                                      | *Image Variation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_variation-versatile_diffusion.py)
| [versatile_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/versatile_diffusion)                 | [**Versatile Diffusion**](https://arxiv.org/abs/2211.08332)                                                      | *Dual Text and Image Guided Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/dual_text_and_image_guided_generation-versatile_diffusion.py)
| [vq_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/vq_diffusion)                 | [**VQ Diffusion**](https://arxiv.org/abs/2111.14822)                                                      | *Text-to-Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-vq_diffusion.py)


**【注意】** Pipelines可以端到端的展示相应论文中描述的扩散模型系统。然而，大多数Pipelines可以使用不同的调度器组件，甚至不同的模型组件。

## Pipelines API

扩散模型系统通常由多个独立训练的模型以及调度器等其他组件构成。
其中每个模型都是在不同的任务上独立训练的，调度器可以很容易地进行替换。
然而，在推理过程中，我们希望能够轻松地加载所有组件并在推理中使用它们，即使某个组件来自不同的库, 为此，所有pipeline都提供以下功能：


- `from_pretrained` 该方法接收PaddleNLP模型库id（例如`runwayml/stable-diffusion-v1-5`）或本地目录路径。为了能够准确加载相应的模型和组件，相应目录下必须提供`model_index.json`文件。

- `save_pretrained` 该方法接受一个本地目录路径，Pipelines的所有模型或组件都将被保存到该目录下。对于每个模型或组件，都会在给定目录下创建一个子文件夹。同时`model_index.json`文件将会创建在本地目录路径的根目录下，以便可以再次从本地路径实例化整个Pipelines。

- `__call__` Pipelines在推理时将调用该方法。该方法定义了Pipelines的推理逻辑，它应该包括预处理、张量在不同模型之间的前向传播、后处理等整个推理流程。


## 任务展示
### 文本图像多模态
<details><summary>&emsp;文图生成（Text-to-Image Generation）</summary>

- stable_diffusion

```python
from ppdiffusers import StableDiffusionPipeline

# 加载模型和scheduler
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 执行pipeline进行推理
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

# 保存图片
image.save("astronaut_rides_horse_sd.png")
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209322401-6ecfeaaa-6878-4302-b592-07a31de4e590.png">
</div>

</details>

<details><summary>&emsp;文本引导的图像放大（Text-Guided Image Upscaling）</summary>

- stable_diffusion_2

```python
from ppdiffusers import StableDiffusionUpscalePipeline
from ppdiffusers.utils import load_image

pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png"
low_res_img = load_image(url).resize((128, 128))

prompt = "a white cat"
upscaled_image = pipe(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("upsampled_cat_sd2.png")
```
<div align="center">
<img alt="image" src="https://user-images.githubusercontent.com/20476674/209324085-0d058b70-89b0-43c2-affe-534eedf116cf.png">
<center>原图像</center>
<img alt="image" src="https://user-images.githubusercontent.com/20476674/209323862-ce2d8658-a52b-4f35-90cb-aa7d310022e7.png">
<center>生成图像</center>
</div>
</details>

<details><summary>&emsp;文本引导的图像编辑（Text-Guided Image Inpainting）</summary>

- stable_diffusion_2

```python
from ppdiffusers import StableDiffusionUpscalePipeline
from ppdiffusers.utils import load_image

pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png"
low_res_img = load_image(url).resize((128, 128))

prompt = "a white cat"
upscaled_image = pipe(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("upsampled_cat_sd2.png")
```
<div align="center">
<img alt="image" src="https://user-images.githubusercontent.com/20476674/209324085-0d058b70-89b0-43c2-affe-534eedf116cf.png">
<center>原图像</center>
<img alt="image" src="https://user-images.githubusercontent.com/20476674/209323862-ce2d8658-a52b-4f35-90cb-aa7d310022e7.png">
<center>生成图像</center>
</div>
</details>


<details><summary>&emsp;文本引导的图像变换（Image-to-Image Text-Guided Generation）</summary>

- stable_diffusion
```python
import paddle

from ppdiffusers import StableDiffusionImg2ImgPipeline
from ppdiffusers.utils import load_image

# 加载pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 下载初始图片
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"

init_image = load_image(url).resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"
# 使用fp16加快生成速度
with paddle.amp.auto_cast(True):
    image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]

image.save("fantasy_landscape.png")
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209327142-d8e1d0c7-3bf8-4a08-a0e8-b11451fc84d8.png">
<center>原图像</center>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209325799-d9ff279b-0d57-435f-bda7-763e3323be23.png">
<center>生成图像</center>
</div>
</details>
</details>

<details><summary>&emsp;文本图像双引导图像生成（Dual Text and Image Guided Generation）</summary>

- versatile_diffusion
```python
from ppdiffusers import VersatileDiffusionDualGuidedPipeline
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg"
image = load_image(url)
text = "a red car in the sun"

pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
pipe.remove_unused_weights()

text_to_image_strength = 0.75
image = pipe(prompt=text, image=image, text_to_image_strength=text_to_image_strength).images[0]
image.save("versatile-diffusion-red_car.png")
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209325965-2475e9c4-a524-4970-8498-dfe10ff9cf24.jpg" >
<center>原图像</center>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209325293-049098d0-d591-4abc-b151-9291ac2636da.png">
<center>生成图像</center>
</div>
</details>

### 图像

<details><summary>&emsp;无条件图像生成（Unconditional Image Generation）</summary>

- latent_diffusion_uncond

```python
from ppdiffusers import LDMPipeline

# 加载模型和scheduler
pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")

# 执行pipeline进行推理
image = pipe(num_inference_steps=200).images[0]

# 保存图片
image.save("ldm_generated_image.png")
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209327936-7fe914e0-0ea0-4e21-a433-24eaed6ee94c.png">
</div>
</details>

<details><summary>&emsp;超分（Super Superresolution）</summary>

- latent_diffusion
```python
import paddle

from ppdiffusers import LDMSuperResolutionPipeline
from ppdiffusers.utils import load_image

# 加载pipeline
pipe = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")

# 下载初始图片
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"

init_image = load_image(url).resize((128, 128))
init_image.save("original-image.png")

# 使用fp16加快生成速度
with paddle.amp.auto_cast(True):
    image = pipe(init_image, num_inference_steps=100, eta=1).images[0]

image.save("super-resolution-image.png")
```
<div align="center">
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209328660-9700fdc3-72b3-43bd-9a00-23b370ba030b.png">
<center>原图像</center>
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209328479-4eaea5d8-aa4a-4f31-aa2a-b47e3c730f15.png">
<center>生成图像</center>
</div>
</details>


<details><summary>&emsp;图像编辑（Image Inpainting）</summary>

- repaint
```python
from ppdiffusers import RePaintPipeline, RePaintScheduler
from ppdiffusers.utils import load_image

img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/celeba_hq_256.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mask_256.png"

# Load the original image and the mask as PIL images
original_image = load_image(img_url).resize((256, 256))
mask_image = load_image(mask_url).resize((256, 256))

scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256", subfolder="scheduler")
pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)

output = pipe(
    original_image=original_image,
    mask_image=mask_image,
    num_inference_steps=250,
    eta=0.0,
    jump_length=10,
    jump_n_sample=10,
)
inpainted_image = output.images[0]

inpainted_image.save("repaint-image.png")
```
<div align="center">
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209329052-b6fc2aaf-1a59-49a3-92ef-60180fdffd81.png">
<center>原图像</center>
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209329048-4fe12176-32a0-4800-98f2-49bd8d593799.png">
<center>mask图像</center>
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209329241-b7e4d99e-468a-4b95-8829-d77ee14bfe98.png">
<center>生成图像</center>
</div>
</details>



<details><summary>&emsp;图像变化（Image Variation）</summary>

- versatile_diffusion
```
from ppdiffusers import VersatileDiffusionImageVariationPipeline
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg"
image = load_image(url)

pipe = VersatileDiffusionImageVariationPipeline.from_pretrained("shi-labs/versatile-diffusion")

image = pipe(image).images[0]
image.save("versatile-diffusion-car_variation.png")
```
<div align="center">
<img  width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209331434-51f6cdbd-b8e4-4faa-8e49-1cc852e35603.jpg">
<center>原图像</center>
<img  width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209331591-f6cc4cd8-8430-4627-8d22-bf404fb2bfdd.png">
<center>生成图像</center>
</div>
</details>





### 音频

<details><summary>&emsp;无条件音频生成（Unconditional Audio Generation）</summary>

- audio_diffusion

```
from scipy.io.wavfile import write
from ppdiffusers import AudioDiffusionPipeline
import paddle

# 加载模型和scheduler
pipe = AudioDiffusionPipeline.from_pretrained("teticio/audio-diffusion-ddim-256")
pipe.set_progress_bar_config(disable=None)
generator = paddle.Generator().manual_seed(42)

output = pipe(generator=generator)
audio = output.audios[0]
image = output.images[0]

# 保存音频到本地
for i, audio in enumerate(audio):
    write(f"audio_diffusion_test{i}.wav", pipe.mel.sample_rate, audio.transpose())

# 保存图片
image.save("audio_diffusion_test.png")
```
<div align = "center">
  <thead>
  </thead>
  <tbody>
   <tr>
      <td align = "center">
      <a href="https://paddlenlp.bj.bcebos.com/models/community/teticio/data/audio_diffusion_test0.wav" rel="nofollow">
            <img align="center" src="https://user-images.githubusercontent.com/20476674/209344877-edbf1c24-f08d-4e3b-88a4-a27e1fd0a858.png" width="200 style="max-width: 100%;"></a><br>
      </td>
    </tr>
  </tbody>
</div>

<div align="center">
<img  width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209342125-93e8715e-895b-4115-9e1e-e65c6c2cd95a.png">
</div>
</details>

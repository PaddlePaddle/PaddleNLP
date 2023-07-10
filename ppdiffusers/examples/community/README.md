# Community Examples

社区示例包含由社区添加的推理和训练示例。可以从下表中了解所有社区实例的概况。点击**Code Example**，跳转到对应实例的可运行代码，可以复制并运行。如果一个示例不能像预期的那样工作，请创建一个issue提问。

|Example|Description|Code Example|Author|
|-|-|-|-|
|CLIP Guided Stable Diffusion|使用CLIP引导Stable Diffusion实现文生图|[CLIP Guided Stable Diffusion](#clip-guided-stable-diffusion)||
|Stable Diffusion Interpolation|在不同的prompts或seed的Stable Diffusion潜空间进行插值|[Stable Diffusion Interpolation](#stable-diffusion-interpolation)||
|Stable Diffusion Mega|一个集成Stable Diffusion 文生图、图生图、图像修复的Pipeline|[Stable Diffusion Mega](#stable-diffusion-mega)||
|Long Prompt Weighting Stable Diffusion| 一个没有token数目限制的Stable Diffusion Pipeline，支持在prompt中解析权重|[Long Prompt Weighting Stable Diffusion](#long-prompt-weighting-stable-diffusion)||
|AUTOMATIC1111 WebUI Stable Diffusion| 与AUTOMATIC1111的WebUI基本一致的Pipeline |[AUTOMATIC1111 WebUI Stable Diffusion](#automatic1111-webui-stable-diffusion)||
|Stable Diffusion with High Resolution Fixing| 使用高分辨率修复功能进行文图生成|[Stable Diffusion with High Resolution Fixing](#stable-diffusion-with-high-resolution-fixing)||
|ControlNet Reference Only| 基于参考图片生成与图片相似的图片|[ControlNet Reference Only](#controlnet-reference-only)||
|Stable Diffusion Mixture Tiling| 基于Mixture机制的多文本大图生成Stable Diffusion Pipeline|[Stable Diffusion Mixture Tiling](#stable-diffusion-mixture-tiling)||
|CLIP Guided Images Mixing Stable Diffusion Pipeline| 一个用于图片融合的Stable Diffusion Pipeline|[CLIP Guided Images Mixing Using Stable Diffusion](#clip-guided-images-mixing-with-stable-diffusion)||

## Example usages

### CLIP Guided Stable Diffusion

使用 CLIP 模型引导 Stable Diffusion 去噪，可以生成更真实的图像。

以下代码运行需要16GB的显存。

```python
import os

import paddle
from clip_guided_stable_diffusion import CLIPGuidedStableDiffusion

from paddlenlp.transformers import CLIPFeatureExtractor, CLIPModel

feature_extractor = CLIPFeatureExtractor.from_pretrained(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                                       dtype=paddle.float32)

guided_pipeline = CLIPGuidedStableDiffusion.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    paddle_dtype=paddle.float16,
)
guided_pipeline.enable_attention_slicing()

prompt = "fantasy book cover, full moon, fantasy forest landscape, golden vector elements, fantasy magic, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Albert Bierstadt, masterpiece"

generator = paddle.Generator().manual_seed(2022)
with paddle.amp.auto_cast(True, level="O2"):
    images = []
    for i in range(4):
        image = guided_pipeline(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            clip_guidance_scale=100,
            num_cutouts=4,
            use_cutouts=False,
            generator=generator,
            unfreeze_unet=False,
            unfreeze_vae=False,
        ).images[0]
        images.append(image)

# save images locally
if not os.path.exists("clip_guided_sd"):
    os.mkdir("clip_guided_sd")
for i, img in enumerate(images):
    img.save(f"./clip_guided_sd/image_{i}.png")
```
生成的图片保存在`images`列表中，样例如下：

|       image_0       |       image_1       |       image_2       |       image_3       |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
|![][clip_guided_sd_0]|![][clip_guided_sd_1]|![][clip_guided_sd_2]|![][clip_guided_sd_3]|

[clip_guided_sd_0]: https://user-images.githubusercontent.com/40912707/220514674-e5cb29a3-b07e-4e8f-a4c8-323b35637294.png
[clip_guided_sd_1]: https://user-images.githubusercontent.com/40912707/220514703-1eaf444e-1506-4c44-b686-5950fd79a3da.png
[clip_guided_sd_2]: https://user-images.githubusercontent.com/40912707/220514765-89e48c13-156f-4e61-b433-06f1283d2265.png
[clip_guided_sd_3]: https://user-images.githubusercontent.com/40912707/220514751-82d63fd4-e35e-482b-a8e1-c5c956119b2e.png

### Wildcard Stable Diffusion

例如我们有下面的prompt:

```python
prompt = "__animal__ sitting on a __object__ wearing a __clothing__"
```
然后，我们可以定义动物、物体和衣服的可能采样值。这些文件可以来自与类别同名的.txt文件。
这些可能值也可以定义为字典，例如：`{"animal":["dog", "cat", mouse"]}`

下面是一个完整的示例：
创建一个`animal.txt`，包含的内容为：

```
dog
cat
mouse
```
创建一个`object.txt`，包含的内容为：
```
chair
sofa
bench
```
代码示例为：
```python
from wildcard_stable_diffusion import WildcardStableDiffusionPipeline

pipe = WildcardStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
)
prompt = "__animal__ sitting on a __object__ wearing a __clothing__"
image = pipe(
    prompt,
    wildcard_option_dict={
        "clothing":["hat", "shirt", "scarf", "beret"]
    },
    wildcard_files=["object.txt", "animal.txt"],
    num_prompt_samples=1
).images[0]
image.save("wildcard_img.png")
```

### Composable Stable diffusion

以下代码需要9GB的显存。
```python
import os

import paddle
from composable_stable_diffusion import ComposableStableDiffusionPipeline

prompt = "mystical trees | A magical pond | dark"
scale = 7.5
steps = 50
weights = "7.5 | 7.5 | -7.5"
pipe = ComposableStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
)
pipe.safety_checker = None

images = []
generator = paddle.Generator().manual_seed(2)
for i in range(4):
    image = pipe(prompt, guidance_scale=scale, num_inference_steps=steps,
                 weights=weights, generator=generator).images[0]
    images.append(image)

# save images locally
if not os.path.exists("composable_sd"):
    os.mkdir("composable_sd")
for i, img in enumerate(images):
    img.save(f"./composable_sd/image_{i}.png")
```

### One Step Unet

one-step-unet可以按照下面的方式运行：

```python
from one_step_unet import UnetSchedulerOneForwardPipeline

pipe = UnetSchedulerOneForwardPipeline.from_pretrained("google/ddpm-cifar10-32")
pipe()
```
这个pipeline不是作为feature使用的，它只是一个如何添加社区pipeline的示例

### Stable Diffusion Interpolation

以下代码运行需要10GB的显存。

```python
from interpolate_stable_diffusion import StableDiffusionWalkPipeline
import paddle

pipe = StableDiffusionWalkPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    paddle_dtype=paddle.float16,
    safety_checker=
    None,  # Very important for videos...lots of false positives while interpolating
    # custom_pipeline="interpolate_stable_diffusion",
)
pipe.enable_attention_slicing()

prompts = [
    'a photo of a landscape in summer',
    'a photo of a landscape in autumn',
]
seeds = [0] * len(prompts)

with paddle.amp.auto_cast(True, level="O2"):
    frame_filepaths = pipe.walk(
        prompts=prompts,
        seeds=seeds,
        num_interpolation_steps=16,
        output_dir='./dreams',
        batch_size=4,
        height=512,
        width=512,
        guidance_scale=8.5,
        num_inference_steps=50,
    )
```

`walk(...)`方法将生成一系列图片，保存在`output_dir`指定的目录下，并返回这些图片的路径。你可以使用这些图片来制造stable diffusion视频。上述代码生成的效果如下：

<center class="half">
<img src="https://user-images.githubusercontent.com/40912707/220613501-df579ae1-c3a3-4f22-8865-d899c4732fe7.gif">
</center>

> 关于如何使用 stable diffusion 制作视频详细介绍以及更多完整的功能，请参考 [https://github.com/nateraw/stable-diffusion-videos](https://github.com/nateraw/stable-diffusion-videos)。


### Stable Diffusion Mega

`StableDiffusionMegaPipeline`可以让你在一个类里使用stable diffusion的主要用例。下述示例代码中展示了在一个pipeline中运行"text-to-image", "image-to-image", and "inpainting"。

```python
from stable_diffusion_mega import StableDiffusionMegaPipeline
import PIL
import requests
from io import BytesIO
import paddle


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


pipe = StableDiffusionMegaPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", paddle_dtype=paddle.float16)
# pipe.to("gpu")
pipe.enable_attention_slicing()
generator = paddle.Generator().manual_seed(2022)

# Text-to-Image
with paddle.amp.auto_cast(True, level="O2"):
    images = pipe.text2img("An astronaut riding a horse",
                           generator=generator).images

images[0].save("text2img.png")

# Image-to-Image

init_image = download_image(
    "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
)
prompt = "A fantasy landscape, trending on artstation"
with paddle.amp.auto_cast(True, level="O2"):
    images = pipe.img2img(prompt=prompt,
                          image=init_image,
                          strength=0.75,
                          guidance_scale=7.5,
                          generator=generator).images
images[0].save("img2img.png")

# Inpainting

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

prompt = "a cat sitting on a bench"
with paddle.amp.auto_cast(True, level="O2"):
    images = pipe.inpaint(prompt=prompt,
                          image=init_image,
                          mask_image=mask_image,
                          strength=0.75,
                          generator=generator).images
images[0].save("inpainting.png")
```
上述代码生成效果如下：

|使用|源|效果|
|:-:|:-:|:-:|
|text-to-image|An astronaut riding a horse|<img src="https://user-images.githubusercontent.com/40912707/220876185-4c2c01f8-90f3-45c4-813a-7143541ec456.png" width="500" />|
|image-to-image|<img src="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg" width="500" /> </br> A fantasy landscape, trending on artstation|<img src="https://user-images.githubusercontent.com/40912707/220876054-5eca5e9a-340e-40a4-a28e-b97af1b006e9.png" width="500" />|
|inpainting|<img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png" width="250" /><img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png" width="250" /> </br>a cat sitting on a bench|<img src="https://user-images.githubusercontent.com/40912707/220876220-ee044a56-6455-4566-9f42-580e29555497.png" width="500" />|

[text2img]: https://user-images.githubusercontent.com/40912707/220876185-4c2c01f8-90f3-45c4-813a-7143541ec456.png
[img2img]: https://user-images.githubusercontent.com/40912707/220876054-5eca5e9a-340e-40a4-a28e-b97af1b006e9.png
[inpainting]: https://user-images.githubusercontent.com/40912707/220876220-ee044a56-6455-4566-9f42-580e29555497.png


### Long Prompt Weighting Stable Diffusion

该自定义Pipeline特征如下：
* 输入提示没有77 token的长度限制
* 包括文生图、图生图、图像修复三种管道
* 给提示片段加上强调，例如 `a baby deer with (big eyes)`
* 给提示片段加上淡化，例如 `a [baby] deer with big eyes`
* 给提示片段加上精确的权重，例如 `a baby deer with (big eyes:1.3)`

prompt加权公示：
* `a baby deer with` == `(a baby deer with:1.0)`
* `(big eyes)` == `(big eyes:1.1)`
* `((big eyes))` == `(big eyes:1.21)`
* `[big eyes]` == `(big eyes:0.91)`

代码示例如下：

```python
import paddle
from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
    "hakurei/waifu-diffusion",
    paddle_dtype=paddle.float16,
    )

prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
neg_prompt = "lowres, bad_anatomy, error_body, error_hair, error_arm, (error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers) error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry"

generator = paddle.Generator().manual_seed(0)

with paddle.amp.auto_cast(True, level="O2"):
    images = pipe.text2img(prompt,
                           negative_prompt=neg_prompt,
                           width=512,
                           height=512,
                           max_embeddings_multiples=3,
                           generator=generator).images

images[0].save("lpw.png")
```

上述代码生成结果如下:

<center><img src="https://user-images.githubusercontent.com/40912707/221503299-24055b14-0b07-4f94-b7f9-d4f84b492540.png" style="zoom:50%"/></center>


### AUTOMATIC1111 WebUI Stable Diffusion
`WebUIStableDiffusionPipeline` 是与 [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 基本对齐的一个pipeline。

该自定义Pipeline支持如下的功能：
* 输入的 token 没有长度限制，可以超过77；
* 支持clip_skip，即可以使用不同层text_encoder的输出；
* 支持直接加载webui中的textual_inversion权重；
* 支持ControlNet；

```python
from pathlib import Path

import cv2
import numpy as np
import paddle
from PIL import Image
from webui_stable_diffusion import WebUIStableDiffusionPipeline

from ppdiffusers import ControlNetModel, DiffusionPipeline
from ppdiffusers.utils import image_grid, load_image

# 支持controlnet模型
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", paddle_dtype=paddle.float16)
pipe = WebUIStableDiffusionPipeline.from_pretrained(
    "TASUKU2023/Chilloutmix", controlnet=controlnet, paddle_dtype=paddle.float16
)
# 或者
# pipe = DiffusionPipeline.from_pretrained("TASUKU2023/Chilloutmix", controlnet=controlnet, paddle_dtype=paddle.float16, custom_pipeline="webui_stable_diffusion")

# 自动下载civitai的lora及ti文件（请注意自己的网络。）
# 介绍网页，程序将自动搜索介绍网页的下载链接
pipe.download_civitai_lora_file("https://civitai.com/models/15365/hanfu")
pipe.download_civitai_lora_file("https://civitai.com/models/12597/moxin")
pipe.download_civitai_ti_file("https://civitai.com/models/1998/autumn-style")
pipe.download_civitai_ti_file("https://civitai.com/models/21131/daisy-ridley-embedding")
# 纯下载链接
pipe.download_civitai_lora_file("https://civitai.com/api/download/models/21656")

print("Supported Lora: " + "、 ".join([p.stem for p in Path(pipe.LORA_DIR).glob("*.safetensors")]))

# 我们需要安装develop版的paddle才可以使用xformers
# pipe.enable_xformers_memory_efficient_attention()
scheduler_name = ["ddim", "pndm", "euler", "dpm-multi"]
for enable_lora in [True, False]:
    images = []
    for sc in scheduler_name:
        # 切换scheduler
        pipe.switch_scheduler(sc)
        # 指定clip_skip
        clip_skip = 1
        # 指定seed
        generator = paddle.Generator().manual_seed(0)
        # guidance_scale
        guidance_scale = 3.5
        prompt = "# shukezouma, negative space, , shuimobysim , portrait of a woman standing , willow branches, (masterpiece, best quality:1.2), traditional chinese ink painting, <lora:MoXinV1:1.0>, modelshoot style, peaceful, (smile), looking at viewer, wearing long hanfu, hanfu, song, willow tree in background, wuchangshuo,"
        negative_prompt = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, skin spots, acnes, skin blemishes, age spot, glans, (watermark:2),"
        img = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            height=768,
            width=512,
            clip_skip=clip_skip,
            guidance_scale=guidance_scale,
            generator=generator,
            enable_lora=enable_lora,
        ).images[0]
        images.append(img)
    if enable_lora:
        image_grid(images, 2, 2).save(f"lora_enable.png")
    else:
        image_grid(images, 2, 2).save(f"lora_disable.png")


image = np.array(
    load_image("https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png")
)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image] * 3, axis=2)
canny_image = Image.fromarray(image)
canny_image = canny_image.resize((512, 768))

# controlnet
for enable_lora in [True, False]:
    images = []
    for sc in scheduler_name:
        pipe.switch_scheduler(sc)
        clip_skip = 1
        generator = paddle.Generator().manual_seed(0)
        guidance_scale = 3.5
        prompt = "a bird <lora:MoXinV1:1.0>"
        negative_prompt = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, skin spots, acnes, skin blemishes, age spot, glans, (watermark:2),"
        img = pipe(
            prompt,
            image=canny_image,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            height=None,  # auto detect image height and width
            width=None,  # auto detect image height and width
            clip_skip=clip_skip,
            guidance_scale=guidance_scale,
            generator=generator,
            enable_lora=enable_lora,
            resize_mode=1,
            controlnet_conditioning_scale=1.0,
        ).images[0]
        images.append(img)
    if enable_lora:
        image_grid(images, 2, 2).save(f"lora_enable_controlnet.png")
    else:
        image_grid(images, 2, 2).save(f"lora_disable_controlnet.png")
```

生成的图片如下所示：
|       lora_disable.png       |       lora_enable.png       |       lora_disable_controlnet.png       |       lora_enable_controlnet.png       |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
|![][lora_disable]|![][lora_enable]|![][lora_disable_controlnet]|![][lora_enable_controlnet]|

[lora_disable]: https://user-images.githubusercontent.com/50394665/230832029-c06a1367-1f2c-4206-9666-99854fcee240.png
[lora_enable]: https://user-images.githubusercontent.com/50394665/230832028-730ce442-dd34-4e36-afd0-81d40843359a.png
[lora_disable_controlnet]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/49ad234e-f92c-4e55-9d4c-86b5d392d704
[lora_enable_controlnet]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/cda43315-cfa5-490a-a2ab-09d9ded7bf44

### Stable Diffusion with High Resolution Fixing
`StableDiffusionHiresFixPipeline` 基于Stable Diffusion进行文图生成，同时启动高分辨率修复功能。该自定义Pipeline生成图像期间共包含两个阶段: 初始生成图像阶段和高清修复阶段。使用方式如下所示：

```python
import paddle
from stable_diffusion_hires_fix import StableDiffusionHiresFixPipeline
from ppdiffusers import EulerAncestralDiscreteScheduler

pipe = StableDiffusionHiresFixPipeline.from_pretrained("stabilityai/stable-diffusion-2", paddle_dtype=paddle.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

generator = paddle.Generator().manual_seed(5232132133)
prompt = "1 real girl, long black hair, detailed face, light smile, chinese style, hanfu"
image = pipe(prompt, guidance_scale=7.5, height=768, width=768, generator=generator, num_inference_steps=40, hires_ratio=0.5, hr_resize_width=768, hr_resize_height=1024, enable_hr=True).images[0]

image.show()

```
生成的图片如下所示：
<center><img src="https://github.com/PaddlePaddle/PaddleNLP/assets/35913314/1c96a219-0b5e-4e1a-b244-0c8cc7cb41f9" width=40%></center>


### ControlNet Reference Only
[Reference-Only Control](https://github.com/Mikubill/sd-webui-controlnet#reference-only-control) 是一种不需要任何控制模型就可以直接使用图像作为参考来引导生成图像的方法。它使用方式如下所示：

```python
import paddle
from reference_only import ReferenceOnlyPipeline
from ppdiffusers import DDIMScheduler
from ppdiffusers.utils import load_image

pipe = ReferenceOnlyPipeline.from_pretrained("TASUKU2023/Chilloutmix", safety_checker=None, paddle_dtype=paddle.float16)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, steps_offset=1, clip_sample=False, set_alpha_to_one=False,)

prompt = "a dog running on grassland, best quality"
input_image = load_image("https://raw.githubusercontent.com/Mikubill/sd-webui-controlnet/main/samples/dog_rel.png")

for control_name in ["none", "reference_only", "reference_adain", "reference_adain+attn"]:
    generator = paddle.Generator().manual_seed(42)
    image = pipe(prompt,
                 guidance_scale=7.,
                 height=512,
                 width=512,
                 image=input_image,
                 num_inference_steps=20,
                 generator=generator,
                 control_name=control_name, # "none", "reference_only", "reference_adain", "reference_adain+attn"
                 attention_auto_machine_weight=1.0, # 0.0~1.0
                 gn_auto_machine_weight=1.0, # 0.0~2.0
                 current_style_fidelity=0.5, # 0.0~1.0
                 resize_mode=0, # ["0 means Just resize", "1 means Crop and resize", "2 means Resize and fill", "-1 means Do nothing"]
                ).images[0]
    image.save(control_name + ".png")
```
生成的图片如下所示：


|       none       |       reference_only       |       reference_adain       |       reference_adain+attn       |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
|![][none]|![][reference_only]|![][reference_adain]|![][reference_adain+attn]|

[none]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/97db3779-9dd7-4d62-ae15-5d2fda68f311
[reference_only]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/4d67e752-cddc-40ab-9524-39e8d9b4a428
[reference_adain]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/266968c7-5065-4589-9bd8-47515d50c6de
[reference_adain+attn]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/73d53a4f-e601-4969-9cb8-e3fdf719ae0c


### Stable Diffusion Mixture Tiling
`StableDiffusionTilingPipeline`是一个基于Mixture机制的多文本大图生成Stable Diffusion Pipeline。使用方式如下所示：

```python
from ppdiffusers import LMSDiscreteScheduler, DiffusionPipeline

# Creater scheduler and model (similar to StableDiffusionPipeline)
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, custom_pipeline="mixture_tiling")

# Mixture of Diffusers generation
image = pipeline(
    prompt=[[
        "A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        "A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        "An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
    ]],
    tile_height=640,
    tile_width=640,
    tile_row_overlap=0,
    tile_col_overlap=256,
    guidance_scale=8,
    seed=7178915308,
    num_inference_steps=50,
)["images"][0]
image.save('mixture_tiling' + ".png")
```
生成的图片如下所示：
<center><img src="https://user-images.githubusercontent.com/20476674/250050184-c3d26d20-dbdf-42f6-9723-5f35f628f68e.png" width=100%></center>

### CLIP Guided Images Mixing With Stable Diffusion
`CLIPGuidedImagesMixingStableDiffusion` 基于Stable Diffusion来针对输入的两个图片进行融合：
```python
import requests
from io import BytesIO

import PIL
import paddle
import open_clip
from open_clip import SimpleTokenizer
from ppdiffusers import DiffusionPipeline
from paddlenlp.transformers import CLIPFeatureExtractor, CLIPModel


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

# Loading additional models
feature_extractor = CLIPFeatureExtractor.from_pretrained(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
)
clip_model = CLIPModel.from_pretrained(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", paddle_dtype=paddle.float16
)

mixing_pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="clip_guided_images_mixing_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    paddle_dtype=paddle.float16,
)
mixing_pipeline.enable_attention_slicing()

# Pipline running
generator = paddle.Generator().manual_seed(17)

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

content_image = download_image("https://paddlenlp.bj.bcebos.com/models/community/westfish/develop/clip_guided_images_mixing_stable_diffusion_images/boromir.jpg")
style_image = download_image("https://paddlenlp.bj.bcebos.com/models/community/westfish/develop/clip_guided_images_mixing_stable_diffusion_images/gigachad.jpg")

pipe_images = mixing_pipeline(
    num_inference_steps=50,
    content_image=content_image,
    style_image=style_image,
    content_prompt="boromir",
    style_prompt="gigachad",
    noise_strength=0.65,
    slerp_latent_style_strength=0.9,
    slerp_prompt_style_strength=0.1,
    slerp_clip_image_style_strength=0.1,
    guidance_scale=9.0,
    batch_size=1,
    clip_guidance_scale=100,
    generator=generator,
).images

pipe_images[0].save('clip_guided_images_mixing_stable_diffusion.png')
```
图片生成效果如下所示：
<div align="center">
<center><img src="https://user-images.githubusercontent.com/20476674/251700919-8abd694f-d93f-4ead-8379-f99405aff1c4.jpg" width=30%></center>
<center>内容图像</center>
<div align="center">
<center><img src="https://user-images.githubusercontent.com/20476674/251700932-4ff5f914-bbd6-4c99-abc4-c7a7fc0fa826.jpg" width=30%></center>
<center>风格图像</center>
<div align="center">
<center><img src="https://user-images.githubusercontent.com/20476674/251701022-c11ea706-f865-4b3f-ab99-9eb79c87439b.png" width=30%></center>
<center>生成图像</center>

# Community Examples

社区示例包含由社区添加的推理和训练示例。可以从下表中了解所有社区实例的概况。点击**Code Example**，跳转到对应实例的可运行代码，可以复制并运行。如果一个示例不能像预期的那样工作，请创建一个issue提问。

|Example|Description|Code Example|Author|
|-|-|-|-|
|CLIP Guided Stable Diffusion|使用CLIP引导Stable Diffusion实现文生图|[CLIP Guided Stable Diffusion](#clip-guided-stable-diffusion)||
|Stable Diffusion Interpolation|在不同的prompts或seed的Stable Diffusion潜空间进行插值|[Stable Diffusion Interpolation](#stable-diffusion-interpolation)||
|Stable Diffusion Mega|一个 Stable Diffusion 管道实现文生图、图生图、图像修复|[Stable Diffusion Mega](#stable-diffusion-mega)||
|Long Prompt Weighting Stable Diffusion| 一个没有token数目限制的Stable Diffusion管道，支持在prompt中解析权重|[Long Prompt Weighting Stable Diffusion](#long-prompt-weighting-stable-diffusion)||


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

```
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
```

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
```
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

```
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

该自定义管线特征如下：
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

上述代码生成结果如下

<center><img src="https://user-images.githubusercontent.com/40912707/221503299-24055b14-0b07-4f94-b7f9-d4f84b492540.png" style="zoom:50%"/></center>

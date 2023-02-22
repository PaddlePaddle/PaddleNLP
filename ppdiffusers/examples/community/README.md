# Community Examples

社区示例包含由社区添加的推理和训练示例。可以从下表中了解所有社区实例的概况。点击**Code Example**，跳转到对应实例的可运行代码，可以复制并运行。如果一个示例不能像预期的那样工作，请创建一个issue提问。

|Example|Description|Code Example|Author|
|-|-|-|-|
|CLIP Guided Stable Diffusion|使用CLIP引导Stable Diffusion实现文生图|[CLIP Guided Stable Diffusion](#CLIP%20Guided%20Stable%20Diffusion)||

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

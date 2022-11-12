# 自定义StableDiffusionInpaintPipeline

```python
import paddle
from io import BytesIO
import json
import requests
import PIL
from ppdiffusers import EulerAncestralDiscreteScheduler
from custom_pipeline import StableDiffusionInpaintPipelineLegacy

eulera = EulerAncestralDiscreteScheduler(
     beta_start=0.00085,
     beta_end=0.012,
     beta_schedule="scaled_linear"
)

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

# 加载已有的noise_file或自己生成
noise_file = 'test_noise_file'
noise = paddle.randn((1, 4, 64, 64))
with open(noise_file, 'w') as f:
    f.write(json.dumps(noise.numpy().tolist()))

pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=eulera, safety_checker=None)
prompt = "a cat sitting on a bench"

# 固定其他部分的随机性
seed = 42
paddle.seed(42)
image= pipe(prompt, noisefile=noise_file, init_image=init_image, mask_image=mask_image,
                  num_images_per_prompt=1, strength=0.75, guidance_scale=7.5, num_inference_steps=26).images[0]
image.save("cat.png")
```
![cat](https://user-images.githubusercontent.com/50394665/201457636-9e710d60-d829-4ad9-ac6e-9e627b7ce661.png)

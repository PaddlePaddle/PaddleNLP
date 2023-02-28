# LORA 使用

## 依赖
- paddlenlp>=2.5.1
- ppdiffusers>=0.11.0
- safetensors
- fastcore

## 使用
可直接加载 https://civitai.com/ 的lora模型（safetensors权重，代码内部会自动权重转换），但是要注意要搭配基础模型。

```python
import lora_helper
from ppdiffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

# 我们加载safetensor版本的权重

# https://civitai.com/models/6779/arknights-texas-the-omertosa
lora_outputs_path = "xarknightsTexasThe_v10.safetensors"

# 加载之前的模型
pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.apply_lora(lora_outputs_path)

prompt               = "A photo of sks dog in a bucket"
negative_prompt      = ""
guidance_scale       = 8
num_inference_steps  = 25
height               = 512
width                = 512

img = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, height=height, width=width, num_inference_steps=num_inference_steps).images[0]
img.save("demo.png")
```

# Stable Diffusion

## 使用 `StableDiffusionAttendAndExcitePipeline` 和 `PNDMScheduler`

给定一个预先训练好的文本到图像的扩散模型（例如Stable Diffusion），方法`Attend-and-Excite`能引导生成模型在图像生成过程中修改交叉注意力的数值，使得生成图片更忠实于输入文本的提示。

使用该pipeline的示例代码如下

```python

from pathlib import Path
import paddle
from ppdiffusers import StableDiffusionAttendAndExcitePipeline, PNDMScheduler


scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=scheduler,
    )

seed = 123
prompt = "A playful kitten chasing a butterfly in a wildflower meadow"
token_indices = [3,6,10]

generator = paddle.Generator().manual_seed(seed)
image = pipe(
    prompt=prompt,
    token_indices=token_indices,
    generator=generator,
).images[0]

# save
output_dir = Path("output_pd")
prompt_output_path = output_dir / prompt
prompt_output_path.mkdir(exist_ok=True, parents=True)
image.save(prompt_output_path / f'{seed}.png')

```

在V100-32GB显卡运行上述代码生成结果如下：

<center>
<img src="https://user-images.githubusercontent.com/40912707/226089491-0f3f66c2-3c88-4518-9ee4-d77debd50e9e.png">
</center>

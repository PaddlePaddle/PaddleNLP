# Stable Diffusion

## Using `StableDiffusionAttendAndExcitePipeline` with `PNDMScheduler`

Given a pre-trained text-to-image diffusion model (e.g., Stable Diffusion) the method, Attend-and-Excite, guides the generative model to modify the cross-attention values during the image synthesis process to generate images that more faithfully depict the input text prompt.

You can run this pipeline as so:

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

And the above script running in V100-32GB generates the following image:

<center>
<img src="https://user-images.githubusercontent.com/40912707/226089491-0f3f66c2-3c88-4518-9ee4-d77debd50e9e.png">
</center>

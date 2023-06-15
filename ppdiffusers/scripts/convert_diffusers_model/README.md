# Stable Diffusion模型转换教程（Pytorch -> Paddle）

本教程支持将Huggingface的[Diffusers](https://github.com/huggingface/diffusers)版本的Stable Diffusion权重转换成[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)版本的Stable Diffusion权重。

Tips：
如果我们想要将原版的权重转换为[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)的权重，我们可以首先使用
Huggingface提供的转换脚本[convert_original_stable_diffusion_to_diffusers.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py)将原版权重转换为[Diffusers](https://github.com/huggingface/diffusers)版本的权重。

## 1 Diffusers 权重转换为 PPDiffusers权重

### 1.1 依赖安装

模型权重转换需要依赖`torch`, `diffusers`, `transformers`, `paddlepaddle`, `paddlenlp`以及`ppdiffusers`，我可使用`pip`执行下面的命令进行快速安装。

```shell
pip install -r requirements.txt
```

### 1.2 模型权重转换

___注意：模型权重转换过程中，需要下载Stable Diffusion模型。为了使用该模型与权重，你必须接受该模型所要求的License，并且获取HF Hub授予的Token。请访问HuggingFace的[model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), 仔细阅读里面的License，然后签署该协议。___

___Tips: Stable Diffusion是基于以下的License: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___

若第一次权重转换模型，需要先登录HuggingFace客户端。执行以下命令进行登录：

```shell
# 安装huggingface_hub
pip install huggingface_hub
# 登录huggingface_hub
huggingface-cli login
```

登录成功后，可执行以下命令行完成模型权重转换。

```shell
python convert_diffusers_stable_diffusion_to_ppdiffusers.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path stable-diffusion-v1-5-ppdiffusers
```

输出的模型目录结构如下：
```shell
├── stable-diffusion-v1-5-ppdiffusers  # 我们指定的输出文件路径
    ├── model_index.json # 模型index文件
    ├── vae # vae权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── text_encoder # text_encoder权重文件夹
        ├── model_config.json
        ├── model_state.pdparams
    ├── unet # unet权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── scheduler # scheduler文件夹
        ├── scheduler_config.json
    ├── feature_extractor # feature_extractor文件夹
        ├── preprocessor_config.json
    ├── safety_checker # safety_checker文件夹
        ├── model_config.json
        ├── model_state.pdparams
    ├── tokenizer # tokenizer文件夹
        ├── tokenizer_config.json
        ├── merges.txt
        ├── special_tokens_map.json
        ├── vocab.json
```

#### 1.3 参数说明

`convert_diffusers_stable_diffusion_to_ppdiffusers.py` 各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|<div style="width: 230pt">--pretrained_model_name_or_path </div> | Huggingface上提供的diffuers版本的diffusion预训练模型。默认为："runwayml/stable-diffusion-v1-5"。更多diffusion预训练模型可参考[CompVis模型列表](https://huggingface.co/CompVis)及[runwayml模型列表](https://huggingface.co/runwayml)，目前仅支持SD版模型。|
|--output_path | 转换后的模型目录。 |


## 2 原版Stable Diffusion模型权重转换为PPDiffusers权重

总共分为2个步骤
- Step1 原版ckpt权重转换为Diffusers权重；
- Step2 Diffusers权重转换为PPDiffusers权重。

### 2.1 依赖安装

模型权重转换需要依赖`omegaconf`, `torch`, `diffusers`, `transformers`, `paddlepaddle`, `paddlenlp`以及`ppdiffusers`，我可使用`pip`执行下面的命令进行快速安装。

```shell
pip install -r requirements.txt
```

### 2.2 模型权重转换

#### Step1 原版ckpt权重转换为Diffusers权重
在开始之前我们需要准备如下的文件：
- Huggingface提供的转换脚本, https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py;
- 原版的权重文件, https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt;
- yaml配置文件, https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml.

所需的文件目录如下所示：
```shell
├── convert_original_stable_diffusion_to_diffusers.py  # Huggingface的转换脚本
├── v1-5-pruned.ckpt # 原版v1-5模型权重文件
├── v1-inference.yaml # yaml配置文件
```

```shell
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path v1-5-pruned.ckpt --original_config_file v1-inference.yaml --dump_path stable-diffusion-v1-5-diffusers
```

输出的模型目录结构如下：

```shell
├── stable-diffusion-v1-5-diffusers  # 我们指定的输出文件路径
    ├── model_index.json # 模型index文件
    ├── vae # vae权重文件夹
        ├── diffusion_pytorch_model.bin
        ├── config.json
    ├── text_encoder # text_encoder权重文件夹
        ├── config.json
        ├── pytorch_model.bin
    ├── unet # unet权重文件夹
        ├── diffusion_pytorch_model.bin
        ├── config.json
    ├── scheduler # scheduler文件夹
        ├── scheduler_config.json
    ├── feature_extractor # feature_extractor文件夹
        ├── preprocessor_config.json
    ├── safety_checker # safety_checker文件夹
        ├── config.json
        ├── pytorch_model.bin
    ├── tokenizer # tokenizer文件夹
        ├── tokenizer_config.json
        ├── merges.txt
        ├── special_tokens_map.json
        ├── vocab.json
```
#### 参数说明

`convert_original_stable_diffusion_to_diffusers.py` 各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|<div style="width: 230pt">--checkpoint_path </div> | 原版Stable Diffusion模型ckpt后缀的权重文件。默认为："v1-5-pruned.ckpt"。更多原版的预训练模型可在[HuggingFace上搜索](https://huggingface.co/)。|
|--original_config_file | 该权重文件所使用的配置文件，默认为"v1-inference.yaml"。 |
|--dump_path | 转换后的Diffusers版本模型目录。 |

#### Step2 Diffusers权重转换为PPDiffusers权重
由于我们已经得到了Huggingface的[Diffusers](https://github.com/huggingface/diffusers)版本的权重，因此我们可以参考第1部分进行权重转换。

我们仅需要运行下面的代码即可成功转换[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)版本的权重。

```shell
python convert_diffusers_stable_diffusion_to_ppdiffusers.py --pretrained_model_name_or_path stable-diffusion-v1-5-diffusers --output_path stable-diffusion-v1-5-ppdiffusers
```

脚本运行完成后，输出的模型目录结构如下：
```shell
├── stable-diffusion-v1-5-ppdiffusers  # 我们指定的输出文件路径
    ├── model_index.json # 模型index文件
    ├── vae # vae权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── text_encoder # text_encoder权重文件夹
        ├── model_config.json
        ├── model_state.pdparams
    ├── unet # unet权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── scheduler # scheduler文件夹
        ├── scheduler_config.json
    ├── feature_extractor # feature_extractor文件夹
        ├── preprocessor_config.json
    ├── safety_checker # safety_checker文件夹
        ├── model_config.json
        ├── model_state.pdparams
    ├── tokenizer # tokenizer文件夹
        ├── tokenizer_config.json
        ├── merges.txt
        ├── special_tokens_map.json
        ├── vocab.json
```





## 3 转换后的权重效果对比

### 3.1 Text-to-Image效果对比
```python
import torch
from diffusers import StableDiffusionPipeline as DiffusersStableDiffusionPipeline
pipe = DiffusersStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
seed = 1024
generator = torch.Generator("cuda").manual_seed(seed)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, generator=generator).images[0]
image.save("diffusers_astronaut_rides_horse.png")
```
![diffusers_astronaut_rides_horse](https://user-images.githubusercontent.com/50394665/201277740-c9b37d59-4ec0-4b3d-8118-bd7f0dfaf352.png)

```python
import paddle
from ppdiffusers import StableDiffusionPipeline as PPDiffusersStableDiffusionPipeline
pipe = PPDiffusersStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
prompt = "a photo of an astronaut riding a horse on mars"
seed = 1024
paddle.seed(seed)
image = pipe(prompt).images[0]
image.save("ppdiffusers_astronaut_rides_horse.png")
```
![ppdiffusers_astronaut_rides_horse](https://user-images.githubusercontent.com/50394665/201277735-fafa458a-9409-4795-887a-897a2851753d.png)

### 3.2 Image-to-Image text-guided generation效果对比
```python
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline as DiffusersStableDiffusionImg2ImgPipeline

pipe = DiffusersStableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image = image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"
seed = 1024
generator = torch.Generator("cuda").manual_seed(seed)
image = pipe(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5, generator=generator).images[0]

image.save("diffusers_fantasy_landscape.png")
```
![diffusers_fantasy_landscape](https://user-images.githubusercontent.com/50394665/201277726-2c2f2fc8-dbfe-4b38-9940-9000bb6c8333.png)

```python
import requests
import paddle
from PIL import Image
from io import BytesIO

from ppdiffusers import StableDiffusionImg2ImgPipeline as PPDiffusersStableDiffusionImg2ImgPipeline

pipe = PPDiffusersStableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image = image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"
seed = 1024
paddle.seed(seed)
image = pipe(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5).images[0]

image.save("ppdiffusers_fantasy_landscape.png")
```
![ppdiffusers_fantasy_landscape](https://user-images.githubusercontent.com/50394665/201277718-f01e8f8d-b560-442f-bf93-c026285c337e.png)
### 3.3 In-painting效果对比
```python
import torch
import PIL
import requests
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline as DiffusersStableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler as DiffusersEulerAncestralDiscreteScheduler

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
scheduler = DiffusersEulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
pipe = DiffusersStableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", scheduler=scheduler)

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
seed = 1024
generator = torch.Generator("cuda").manual_seed(seed)
image = pipe(prompt=prompt, image=image, mask_image=mask_image, generator=generator).images[0]

image.save("diffusers_cat_on_bench.png")
```
![diffusers_cat_on_bench](https://user-images.githubusercontent.com/50394665/201277724-76145ee6-a3ef-49e7-a1e9-8ccd3c9eb39e.png)

```python
import paddle
import PIL
import requests
from io import BytesIO

from ppdiffusers import StableDiffusionInpaintPipeline as PPDiffusersStableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler as PPDiffusersEulerAncestralDiscreteScheduler

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
scheduler = PPDiffusersEulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
pipe = PPDiffusersStableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", scheduler=scheduler)

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
seed = 1024
paddle.seed(seed)
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

image.save("ppdiffusers_cat_on_bench.png")
```
![ppdiffusers_cat_on_bench](https://user-images.githubusercontent.com/50394665/201277712-2e10c188-e1ca-44f5-b963-657e9d51cc95.png)

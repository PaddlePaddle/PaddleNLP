# LDM权重转换脚本
本目录下包含了两个脚本文件：
- **convert_orig_ldm_ckpt_to_ppdiffusers.py**: LDM原版Pytorch权重转换为PPDiffusers版LDM权重。
- **convert_ppdiffusers_to_orig_ldm_ckpt.py**: PPDiffusers版的LDM权重转换为原版LDM权重。

## 1. LDM原版Pytorch权重转换为PPDiffusers版LDM权重
### 1.1 转换权重
假设已经有了原版权重`"ldm_1p4b_init0.ckpt"`
```bash
python convert_orig_ldm_ckpt_to_ppdiffusers.py \
    --checkpoint_path ldm_1p4b_init0.ckpt \
    --dump_path ldm_1p4b_init0_pytorch \
    --original_config_file text2img_L32H1280_unet800M.yaml
```

### 1.2 推理预测
```python
import paddle
from ppdiffusers import LDMTextToImagePipeline
model_path = "./ldm_1p4b_init0_pytorch"
pipe = LDMTextToImagePipeline.from_pretrained(model_path)
prompt = "a blue tshirt"
image = pipe(prompt, guidance_scale=7.5)[0][0]
image.save("demo.jpg")
```

## 2. PPDiffusers版的LDM权重转换为原版LDM权重
### 2.1 转换权重
假设我们已经使用 `../generate_pipelines.py`生成了`ldm_pipelines`目录。
```shell
├── ldm_pipelines  # 我们指定的输出文件路径
    ├── model_index.json # 模型index文件
    ├── vqvae # vae权重文件夹！实际是vae模型，文件夹名字与HF保持了一致！
        ├── model_state.pdparams
        ├── config.json
    ├── bert # ldmbert权重文件夹
        ├── model_config.json
        ├── model_state.pdparams
    ├── unet # unet权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── scheduler # ddim scheduler文件夹
        ├── scheduler_config.json
    ├── tokenizer # bert tokenizer文件夹
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        ├── vocab.txt
```

```bash
python convert_ppdiffusers_to_orig_ldm_ckpt.py \
    --model_name_or_path ./ldm_pipelines \
    --dump_path ldm_19w.ckpt
```

### 2.2 推理预测
使用`CompVis`[原版txt2img.py](https://github.com/CompVis/latent-diffusion/blob/main/scripts/txt2img.py)脚本生成图片。
```shell
python ./txt2img.py --prompt "a blue t shirt" --ddim_eta 0.0 --n_samples 1 --n_iter 1 --scale 7.5  --ddim_steps 50
```

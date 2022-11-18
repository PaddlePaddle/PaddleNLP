# LDM原版Pytorch权重转换为PPDiffusers权重

## 1. 转换权重
假设已经有了原版权重`"ldm_1p4b_init0.ckpt"`
```bash
python convert_orig_ldm_ckpt_to_ppdiffusers.py \
    --checkpoint_path ldm_1p4b_init0.ckpt \
    --dump_path ldm_1p4b_init0_pytorch \
    --original_config_file text2img_L32H1280_unet800M.yaml
```

## 2. 推理预测
```python
import paddle
from ppdiffusers import LDMTextToImagePipeline
model_path = "./ldm_1p4b_init0_pytorch"
pipe = LDMTextToImagePipeline.from_pretrained(model_path)
prompt = "a blue tshirt"
image = pipe(prompt, guidance_scale=7.5)[0][0]
image.save("demo.jpg")
```

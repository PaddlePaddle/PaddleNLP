# 将原版pytorch权重转换为ppdiffusers预训练权重

## Step1
假设已经有了原版权重`"ldm_1p4b_init0.ckpt"`，从中提取出3个pt文件，ldmbert，unet，vae。
```bash
python extract_orig_weights.py
```

## Step2
转换ldmbert权重(12层)
```bash
python convert_orig_ldmbert_to_ppdiffusers.py
```

## Step3
转换unet和vae权重
```bash
python convert_orig_unet_vae_to_ppdiffusers.py --original_config_file text2img_L12H768_unet800M.yaml
```

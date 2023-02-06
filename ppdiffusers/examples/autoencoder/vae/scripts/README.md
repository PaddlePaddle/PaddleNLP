# 脚本文件

本目录下包含了四个脚本文件：
- **convert_kl_8_to_ppdiffusers.py**: 将原版LDM中的 VAE 权重转换为 Paddle 版的权重，注意：我们转换过程中同时转换了 loss 部分的权重。
- **get_autoencoder_results.py**: 加载训练好的模型文件，然后生成待评估的图片。
- **fid_score.py**: 计算 fid score 的代码。
- **calculate_psnr_ssim.py**: 计算 psnr 和 ssim 指标的代码。

## 1. Pytorch权重转换为Paddle权重
假设我们已经预先使用原版LDM代码初始化了一个`"ldm_vae_init0.ckpt"`权重。然后我们需要使用下面的代码进行权重转换。

```shell
python convert_kl_8_to_ppdiffusers.py \
    --checkpoint_path ldm_vae_init0.ckpt \
    --dump_path ldm_vae_init0_paddle \
    --original_config_file ../config/f8encoder_f16decoder.yaml
```
经过转换后，我们可以得到下面的目录结构。

```shell
├── ldm_vae_init0_paddle  # 我们指定的输出文件路径
    ├── model_state.pdparams
    ├── config.json
```

## 2. 评估训练好的模型性能

### 2.1 生成待评估的图片

```shell
python get_autoencoder_results.py --vae_path "./autoencoder_outputs/checkpoint-200000" --src_size 256 --tgt_size 512 --imgs './coco_val2014_resize512_centercrop/*.png' --outdir generate_images/
```

### 2.2 计算FID指标

```shell
python get_autoencoder_results.py --src_size 256 --tgt_size 512 --imgs './coco_val2014_resize512_centercrop/*.png' --outdir generate_images/
```

### 2.3 计算PSNR和SSIM指标

```shell
python fid_score.py ./coco_val2014_resize512_centercrop/ ./generate_images/ --device gpu

python calculate_psnr_ssim.py --imgs1 'coco_val2014_resize512_centercrop/*.png' --imgs2 'generate_images/*.png'
```

# VAE (KL-8.ckpt)转换脚本

本目录下包含了一个脚本文件：
- **convert_kl_8_to_ppdiffusers.py**: 将原版LDM中的VAE权重转换为Paddle版的权重，注意：我们转换过程中同时转换了loss部分的权重。

## 转换权重
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

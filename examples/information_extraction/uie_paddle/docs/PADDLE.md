## Convert Torch to Paddle

### Convert UIE Model
``` bash
source_torch_folder=hf_models/uie-base-en/
target_paddle_folder=pd_models/uie-base-en

cp -r etc/uie_paddle_tokenizer ${target_paddle_folder}
python scripts/convert_pytorch_to_paddle.py \
  --pytorch_checkpoint_path ${source_torch_folder} \
  --paddle_dump_path ${target_paddle_folder}
```

### Convert T5 Char Model
``` bash
source_torch_folder=hf_models/t5-char-100g-small-30w-uie-zh-qb-50w
target_paddle_folder=pd_models/t5-char-100g-small-30w-uie-zh-qb-50w

cp -r etc/t5_char_paddle_tokenizer ${target_paddle_folder}
python scripts/convert_pytorch_to_paddle.py \
  --pytorch_checkpoint_path ${source_torch_folder} \
  --paddle_dump_path ${target_paddle_folder}
```

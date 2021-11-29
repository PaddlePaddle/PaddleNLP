# Convert tp paddle
1. download checkpoints from huggingface model hub
2. modify model path and run script blow.
```bash
# path1: sshleifertiny-distilbert-base-uncased-finetuned-sst-2-english
# path2: distilbert-base-multilingual-cased
export path1='sshleifertiny-distilbert-base-uncased-finetuned-sst-2-english/pytorch_model.bin'
export path2='distilbert-base-multilingual-cased/pytorch_model.bin'
python convert_to_paddle.py \
--sshleifertiny_model_path $path1 \
--base_model_path $path2
```

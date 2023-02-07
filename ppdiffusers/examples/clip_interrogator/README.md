# clip-interrogator

## 依赖
```shell
pip install -r requirements.txt

# 如果paddlenlp 2.5.1还没有发布，那么需要安装develop版本的paddlenlp
pip install paddlenlp>=2.5.1
```

## 使用

### 快速开始
```python
from PIL import Image
from clip_interrogator import Config, Interrogator
image = Image.open(image_path).convert('RGB')
ci = Interrogator(Config(clip_pretrained_model_name_or_path="openai/clip-vit-large-patch14"))
print(ci.interrogate(image))
```

### Gradio
```shell
python run_gradio.py \
    --clip="openai/clip-vit-large-patch14" \
    --blip="Salesforce/blip-image-captioning-large" \
    --share
```

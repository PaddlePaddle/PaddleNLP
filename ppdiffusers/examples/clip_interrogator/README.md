# clip-interrogator

## 依赖
```shell
pip install -r requirements.txt

```
## 准备data数据（包含artists.txt、flavors.txt、mediums.txt、movements.txt）
```shell
wget https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip-image-captioning-large/data.zip
# 将data文件解压至clip_interrogator目录下
unzip -d clip_interrogator data.zip
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

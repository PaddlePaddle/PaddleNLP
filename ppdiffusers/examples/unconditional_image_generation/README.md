## 训练样本

### 安装依赖

运行脚本之前，请确保安装库的训练依赖：


切换到 example 目录并且运行：
```bash
pip install -r requirements.txt
```


### Unconditional Flowers

下面的命令是使用Oxford Flowers dataset来训练一个DDPM UNet模型：

```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3"  train_unconditional.py \
  --dataset_name="huggan/flowers-102-categories" \
  --cache_dir 'data' \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-flowers-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500
```

完整的训练需要在4xV100 GPUs上训练2小时.

<img src="https://user-images.githubusercontent.com/26864830/180248660-a0b143d0-b89a-42c5-8656-2ebf6ece7e52.png" width="700" />


### Unconditional Pokemon

下面的命令是Pokemon dataset上训练一个DDPM UNet模型：

```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_unconditional.py \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-pokemon-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500
```

完整的训练需要在4xV100 GPUs上训练2小时.

<img src="https://user-images.githubusercontent.com/26864830/180248200-928953b4-db38-48db-b0c6-8b740fe6786f.png" width="700" />


### 使用你自己的数据



要使用自己的数据集，有两种方法：

-您可以将自己的文件夹提供为`--train_data_dir`

-或者，您可以将数据集上传到hub，然后简单地传递`--dataset_name`参数。

下面，我们将对两者进行更详细的解释。

#### 将数据集作为文件夹提供

如果为自己的文件夹提供图像，脚本需要以下目录结构:

```bash
data_dir/xxx.png
data_dir/xxy.png
data_dir/[...]/xxz.png
```

换句话说，脚本将负责收集文件夹中的所有图像。然后可以像这样运行脚本:

```bash
python train_unconditional.py \
    --train_data_dir <path-to-train-directory> \
    <other-arguments>
```

这个脚本将会使用 [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) 特征，并且自动把这些目录变成Dataset对象。

#### 把你的数据上传到hub上

使用[`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder)中提供的功能将图像数据集上传到hub中心非常容易。只需执行以下操作:

```python
from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("imagefolder", data_dir="path_to_your_folder")

# example 2: local files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="path_to_zip_file")

# example 3: remote files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip")

# example 4: providing several splits
dataset = load_dataset("imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]})
```

`ImageFolder将创建包含PIL编码图像的“image”列。

下一步，将数据集推到hub上

```python
# assuming you have ran the huggingface-cli login command in a terminal
dataset.push_to_hub("name_of_your_dataset")

# if you want to push to a private repo, simply pass private=True:
dataset.push_to_hub("name_of_your_dataset", private=True)
```

就这样！现在，只需将“--dataset_name”参数设置为hub上数据集的名称，即可训练模型。

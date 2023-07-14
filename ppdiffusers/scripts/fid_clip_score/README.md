# FID score for PaddlePaddle

FID（Frechet Inception Distance score，FID）是计算真实图像和生成图像的特征向量之间距离的一种度量，最常用于评估生成性对抗网络样本的质量。FID 从原始图像的计算机视觉特征的统计方面的相似度来衡量两组图像的相似度，这种视觉特征是使用 `Inception v3` 图像分类模型计算的得到的。分数越低代表两组图像越相似，或者说二者的统计量越相似，FID 在最佳情况下的得分为 0.0，表示两组图像相同。


## 依赖

- PaddlePaddle
- Pillow
- Numpy
- Scipy

## 快速使用

计算两个图片数据集的FID，`path/to/dataset1`/`path/to/dataset2`为图片文件夹
```
python fid_score.py path/to/dataset1 path/to/dataset2
```

使用CPU计算
```
python fid_score.py path/to/dataset1 path/to/dataset2 --device cpu
```

参数说明
- `batch-size`：使用批次的大小，默认为50
- `num-workers`： 用于加载数据的子进程个数，默认为`min(8, num_cpus)`。
- `device`：使用设备，支持GPU、CPU。
- `dims`：要使用的Inception特征的维度。默认使用2048.

## 参考

- [https://github.com/mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)

# 在COCO英文1k（或30k）数据集上评估 FID score 和 Clip Score指标

```shell
├── outputs
    ├── mscoco.en_g3 # guidance_scales为3的输出图片
        ├── 00000_000.png
        ├── 00001_000.png
        ......
        ├── 00999_000.png
    ├── mscoco.en_g4 # guidance_scales为4的输出图片
        ├── 00000_000.png
        ├── 00001_000.png
        ......
        ├── 00999_000.png
    ......
    ├── mscoco.en_g8 # guidance_scales为8的输出图片
        ├── 00000_000.png
        ├── 00001_000.png
        ......
        ├── 00999_000.png
```
假设我们已经有了上述目录结构的图片，那么我们可以使用`compute_fid_clip_score.py`计算fid score和clip score两个指标。

```shell
python compute_fid_clip_score.py \
    --image_path outputs/mscoco.en_g3 outputs/mscoco.en_g4 outputs/mscoco.en_g5 outputs/mscoco.en_g6 outputs/mscoco.en_g7 outputs/mscoco.en_g8 \
    --text_file_name coco30k \
    --clip_model_name_or_path openai/clip-vit-base-patch32 \
    --resolution 256 \
    --fid_batch_size 32 \
    --clip_batch_size 64 \
    --device gpu
```

参数说明
- `image_path`：我们需要评估的图片文件夹地址，两个地址之间需要用空格分隔。
- `text_file_name`： clip评估所需要的文件名称，可从`["coco1k", "coco10k", "coco30k"]`选择，1k表示1k图片，30k表示30k图片。
- `clip_model_name_or_path`：clip评估所使用的模型。
- `resolution`：fid评估时候所使用的图片的分辨率。
- `fid_batch_size`：fid评估时候所使用的批次。
- `clip_batch_size`：clip评估时候所使用的批次。
- `device`：使用设备，支持GPU、CPU，如"cpu", "gpu:0", "gpu:1"。

![ddim-19w-30k-256](https://user-images.githubusercontent.com/50394665/203267067-6367d675-8580-4c3e-90b0-d8c1ed0d58aa.png)

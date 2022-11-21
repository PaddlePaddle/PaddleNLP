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

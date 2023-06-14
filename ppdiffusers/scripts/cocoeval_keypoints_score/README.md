# COCOeval for Keypints

本文档将指导你如何利用`pycocotools.cocoeval.COCOeval`针对关键点控制生成任务进行控制效果评估。这个工具提供的评估代码可被用于公开的COCO验证集或任何采用相同格式的数据集。它能计算以下几种指标。为了获取合适的格式的测试数据，我们需要运行`get_openpose_keypoints_result_coco_format.py`脚本，它将对真实图像集合和生成的图像集合进行关键点提取，并生成目标格式的关键点检测文件。

评估关键点检测的核心理念是模仿用于目标检测的评价指标，即平均精度（AP）和平均召回率（AR）及其变体。这些指标的核心是真实目标和预测目标之间的相似性度量。在目标检测的下，交并比（IoU）就充当了这种相似性度量（适用于框和段）。通过设定IoU阈值，定义真实目标和预测目标之间的匹配，从而能够计算精度-召回率曲线。为了将AP/AR应用于关键点检测，我们只需要定义一个类似的相似性度量。我们通过定义目标关键点相似度（OKS）来实现这一点，它起着与IoU相同的作用。

具体来说，以下10个具体指标用于描述关键点检测的效果，其中第一指标最为关键：
```
Average Precision (AP):
AP
% AP at OKS=.50:.05:.95 (primary challenge metric)

APOKS=.50
% AP at OKS=.50 (loose metric)

APOKS=.75
% AP at OKS=.75 (strict metric)

AP Across Scales:
APmedium
% AP for medium objects: 322 < area < 962

APlarge
% AP for large objects: area > 962

Average Recall (AR):
AR
% AR at OKS=.50:.05:.95

AROKS=.50
% AR at OKS=.50

AROKS=.75
% AR at OKS=.75

AR Across Scales:
ARmedium
% AR for medium objects: 322 < area < 962

ARlarge
% AR for large objects: area > 962

```


## 依赖
- pycocotools


## 使用方法

首先，我们需要预备原始图片数据集，位置为`path/to/images_origin`。此外，我们还需要准备如`path/to/images_generate1`、`path/to/images_generate2`等待测试的生成图片数据集。执行以下步骤，我们可以得到`xx_gt.json`和`xx_dt.json`：
```
python get_openpose_keypoints_result_coco_format.py \
    --do_gt \
    path/to/images_origin \
    path/to/output/images_origin_gt.json \
    path/to/output/images_origin_ppdet
```
```
python get_openpose_keypoints_result_coco_format.py \
    path/to/images_generate1 \
    path/to/output/images_generate1_dt.json \
    path/to/output/images_generate1_ppdet
python get_openpose_keypoints_result_coco_format.py \
    path/to/images_generate2 \
    path/to/output/images_generate2_dt.json \
    path/to/output/images_generate2_ppdet
```

其次我们需要执行以下命令来获取具体的测试指标：
```
python cocoeval_keypoints.py \
    --gt path/to/output/images_origin_gt.json \
    --dt path/to/output/images_generate1_dt.json
python cocoeval_keypoints.py \
    --gt path/to/output/images_origin_gt.json \
    --dt path/to/output/images_generate2_dt.json
```

## 参考

- [https://cocodataset.org/#keypoints-eval](https://cocodataset.org/#keypoints-eval)

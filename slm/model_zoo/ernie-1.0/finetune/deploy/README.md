# ERNIE 1.0 模型 Python 推理示例

本目录下提供 `infer.py` 快速完成在 CPU/GPU 的中文情感分类任务的 Python 推理示例。

## 快速开始

以下示例展示 ERNIE 1.0 模型在 ChnSenticorp 数据集上进行文本分类任务的 Python 预测部署，可通过命令行参数`--device`指定运行在不同的硬件，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照 [ERNIE 1.0 训练文档](../../README.md)导出得到的部署模型，其模型目录为`model_zoo/ernie-1.0/finetune/tmp/export`（用户可按实际情况设置）。


```bash
# CPU 推理
python infer.py --model_dir ../tmp/chnsenticorp_v2/export/ --device cpu
# GPU 推理
python infer.py --model_dir ../tmp/chnsenticorp_v2/export/ --device gpu
```

运行完成后返回的结果如下：

```bash
......
Batch id: 1189, example id: 0, sentence: 作为五星级 酒店的硬件是差了点 装修很久 电视很小 只是位置很好 楼下是DFS 对面是海港城 但性价比不高, label: positive, negative prob: 0.0001, positive prob: 0.9999.
Batch id: 1190, example id: 0, sentence: 最好别去,很差,看完很差想换酒店,他们竟跟我要服务费.也没待那房间2分种,居然解决了问题,可觉的下次不能去的,, label: negative, negative prob: 1.0000, positive prob: 0.0000.
Batch id: 1191, example id: 0, sentence: 看了一半就看不下去了，后半本犹豫几次都放下没有继续看的激情，故事平淡的连个波折起伏都没有，职场里那点事儿也学得太模糊，没有具体描述，而且杜拉拉就做一个行政而已，是个人都会做的没有技术含量的工作 也能描写的这么有技术含量 真是为难作者了本来冲着畅销排行第一买来看看，觉得总不至于大部分人都没品味吧？结果证明这个残酷的事实，一本让人如同嚼蜡的“畅销书”......, label: negative, negative prob: 0.9999, positive prob: 0.0001.
Batch id: 1192, example id: 0, sentence: 酒店环境很好 就是有一点点偏 交通不是很便利 去哪都需要达车 关键是不好打 酒店应该想办法解决一下, label: positive, negative prob: 0.0003, positive prob: 0.9997.
Batch id: 1193, example id: 0, sentence: 价格在这个地段属于适中, 附近有早餐店,小饭店, 比较方便,无早也无所, label: positive, negative prob: 0.1121, positive prob: 0.8879.
Batch id: 1194, example id: 0, sentence: 酒店的位置不错，附近都靠近购物中心和写字楼区。以前来大连一直都住，但感觉比较陈旧了。住的期间，酒店在进行装修，翻新和升级房间设备。好是好，希望到时房价别涨太多了。, label: positive, negative prob: 0.0000, positive prob: 1.0000.
Batch id: 1195, example id: 0, sentence: 位置不很方便，周围乱哄哄的，卫生条件也不如其他如家的店。以后绝不会再住在这里。, label: negative, negative prob: 1.0000, positive prob: 0.0000.
Batch id: 1196, example id: 0, sentence: 抱着很大兴趣买的，买来粗粗一翻排版很不错，姐姐还说快看吧，如果好我也买一本。可是真的看了，实在不怎么样。就是中文里夹英文单词说话，才翻了2页实在不想勉强自己了。我想说的是，练习英文单词，靠这本书肯定没有效果，其它好的方法比这强多了。, label: negative, negative prob: 1.0000, positive prob: 0.0000.
Batch id: 1197, example id: 0, sentence: 东西不错，不过有人不太喜欢镜面的，我个人比较喜欢，总之还算满意。, label: positive, negative prob: 0.0001, positive prob: 0.9999.
Batch id: 1198, example id: 0, sentence: 房间不错,只是上网速度慢得无法忍受,打开一个网页要等半小时,连邮件都无法收。另前台工作人员服务态度是很好，只是效率有得改善。, label: positive, negative prob: 0.0001, positive prob: 0.9999.
......
```

## 参数说明

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录， |
|--batch_size |输入的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--device_id | 运行设备的 id。默认为0。 |
|--cpu_threads | 当使用 cpu 推理时，指定推理的 cpu 线程数，默认为1。|

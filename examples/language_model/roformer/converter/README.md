## 预训练模型转换

预训练模型可以从 huggingface/transformers 转换而来，方法如下（适用于roformer模型，其他模型按情况调整）：

1. 从huggingface.co获取roformer模型权重
2. 设置参数运行convert.py代码
3. 例子：
   假设我想转换https://huggingface.co/junnyu/roformer_chinese_base 权重
   - (1)首先下载 https://huggingface.co/junnyu/roformer_chinese_base/tree/main 中的pytorch_model.bin文件,假设我们存入了`./roformer_chinese_base/pytorch_model.bin`
   - (2)运行convert.py
        ```bash
        python convert.py \
            --pytorch_checkpoint_path ./roformer_chinese_base/pytorch_model.bin \
            --paddle_dump_path ./roformer_chinese_base/model_state.pdparams
        ```
   - (3)最终我们得到了转化好的权重`./roformer_chinese_base/model_state.pdparams`

## 预训练MLM测试
    ```bash
    python test_mlm.py --model_name roformer-chinese-base --text 今天[MASK]很好，我想去公园玩！
    # paddle: 今天[天气||天||阳光||太阳||空气]很好，我想去公园玩！
    python test_mlm.py --model_name roformer-chinese-base --text 北京是[MASK]的首都！
    # paddle: 北京是[中国||谁||中华人民共和国||我们||中华民族]的首都！
    python test_mlm.py --model_name roformer-chinese-char-base --text 今天[MASK]很好，我想去公园玩！
    # paddle: 今天[天||气||都||风||人]很好，我想去公园玩！
    python test_mlm.py --model_name roformer-chinese-char-base --text 北京是[MASK]的首都！
    # paddle: 北京是[谁||我||你||他||国]的首都！
    ```

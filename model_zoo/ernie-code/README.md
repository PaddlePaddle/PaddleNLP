# ERNIE-CODE

## 快速开始

本项目是ERINE-CODE模型的 PaddlePaddle 实现， 包含模型预测,权重转化。下是本例的简要目录结构及说明：

```text
├── converter.py            # 权重转化脚本
├── predict.py              # 前向预测示例demo
├── README.md               # 文档
```

### 文本生成

本项目提供了简单的文本生成的demo，启动方式如下：

```shell
python predict.py \
  --input 'BadZipFileのAliasは、古い Python バージョンとの互換性のために。' \
  --target_lang 'code' \
  --source_prefix 'translate Japanese to Python: \n' \
  --max_length 1024 \
  --num_beams 3 \
  --device 'gpu'
```

配置文件中参数释义如下：
- `input`: 表示输入的文本序列。
- `target_lang`: 表示目标语言，可指定为'text'或'code'。
- `source_prefix`: 表示提示词。
- `max_length`: 表示输入/输出文本最大长度。
- `num_beams`: 表示解码时每个时间步保留的beam-size。
- `device`: 表示运行设备，可设置为'cpu'或'gpu'。


生成效果展示:
```text
text：BadZipFileのAliasは、古い Python バージョンとの互換性のために。？
code：def badzip_file_alias(self): \n        """BadZipFileのAliasは、古いPython バージョンとの互換性のために。 \n        """ \n        if self._version == \'1.0\': \n            return self._version \n        else: \n            return self._version
```

## 模型导出预测

本项目提供了权重转化脚本`converter.py`，用户可以参考该脚本将Huggingface模型权重转化为paddle形式。

```"shell
python converter.py \
  --pytorch_checkpoint_path /home/models/pytorch_model.bin \
  --paddle_dump_path /home/models/model_state.pdparams
```

## 参考文献
- [ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for Programming Languages](https://arxiv.org/abs/2212.06742)

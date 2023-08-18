# LLaMA Inference

## 动态图推理
默认按mp8执行，如需单卡执行，需修改--gpus参数
```
bash run.sh {input_model_dir}
```

## 动转静导出
默认按mp8执行，如需单卡执行，需修改--gpus参数
```
bash export.sh.sh {input_model_dir} {inference_model_dir}
```

## 静态图推理
默认按mp8执行，如需单卡执行，需修改--gpus参数
```
bash inference.sh {inference_model_dir}
```

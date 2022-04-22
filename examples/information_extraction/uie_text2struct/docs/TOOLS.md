# Tools

### Convert PyTorch Checkpoint to Paddle
PyTorch Checkpoint 转换为 Paddle 版 (convert_pytorch_to_paddle.py)
``` text
 $ python scripts/convert_pytorch_to_paddle.py -h  
usage: convert_pytorch_to_paddle.py [-h] [--pytorch_checkpoint_path PYTORCH_CHECKPOINT_PATH] [--paddle_dump_path PADDLE_DUMP_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --pytorch_checkpoint_path PYTORCH_CHECKPOINT_PATH
                        Path to the Pytorch checkpoint path.
  --paddle_dump_path PADDLE_DUMP_PATH
                        Path to the output Paddle model.
```

### Evalaute Model Performance
验证模型性能 (eval_extraction.py)
```text
 $ python scripts/eval_extraction.py -h  
usage: eval_extraction.py [-h] [-g GOLD_FOLDER] [-p PRED_FOLDER [PRED_FOLDER ...]] [-v] [-w] [-m] [-case]

optional arguments:
  -h, --help            show this help message and exit
  -g GOLD_FOLDER        Golden Dataset folder
  -p PRED_FOLDER [PRED_FOLDER ...]
                        Predicted model folder
  -v                    Show more information during running
  -w                    Write evaluation results to predicted folder
  -m                    Match predicted result multiple times
  -case                 Show case study
```

### Check offset mapping performance
验证回标的性能 (check_offset_map_gold_as_pred.bash)
``` bash
bash scripts/check_offset_map_gold_as_pred.bash <data-folder> <map-config>
```

### SEL-To-Record convertor
将结构化表达式转换成 Record 结构 (sel2record.py)
``` text
 $ python scripts/sel2record.py -h  
usage: sel2record.py [-h] [-g GOLD_FOLDER] [-p PRED_FOLDER [PRED_FOLDER ...]] [-c MAP_CONFIG] [-d DECODING] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -g GOLD_FOLDER        Gold Folder
  -p PRED_FOLDER [PRED_FOLDER ...]
                        Pred Folder
  -c MAP_CONFIG, --config MAP_CONFIG
                        Offset Mapping Config
  -d DECODING
  -v, --verbose         More details information.
```

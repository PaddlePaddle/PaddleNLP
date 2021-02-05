## 预训练模型转换

预训练模型可以从 huggingface/transformers 转换而来，方法如下（适用于bert模型，其他模型按情况调整）：

1. 获取 huggingface/transformers repo
   ```shell
   git clone https://github.com/huggingface/transformers.git
   # 此前操作验证版本为 v2.5.1
   git fetch orgin v2.5.1:v2.5.1
   git checkout v2.5.1
   ```
2. 修改 huggingface/transformers 相关代码并运行:
    ```shell
    export CONVERT_MODEL=$PWD # convert_model所在路径
    export HUGGINGFACE=$PWD/transformers # git clone的transformers所在路径
    cp $CONVERT_MODEL/modeling_utils.py $HUGGINGFACE/src/transformers/modeling_utils.py # 在470行后加入了numpy转换与保存代码
    cp $CONVERT_MODEL/modeling_bert.py  $HUGGINGFACE/src/transformers/modeling_bert.py # 加入了print(outputs)，方便验证对齐转换结果
    cp $CONVERT_MODEL/run_glue.py $HUGGINGFACE/examples/run_glue.py # 去掉数据随机性，换成eval模式，使用pretraining的模型替换下游任务的模型
    ```

    下载QNLI数据：
    ```shell
    wget https://dataset.bj.bcebos.com/glue/QNLI.zip
    unzip QNLI
    ```

    Python3环境中运行：
    ```shell
    export CUDA_VISIBLE_DEVICES=0
    export DATA_DIR=$PWD/QNLI # 上一步中QNLI所在路径
    export MODEL=bert-base-uncased # 希望转换的模型
    export PYTHONPATH=$PYTHONPATH:$HUGGINGFACE/src

    python $HUGGINGFACE/examples/run_glue.py \
        --model_type bert \
        --model_name_or_path $MODEL \
        --task_name QNLI \
        --do_train \
        --do_eval \
        --data_dir $DATA_DIR \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=32   \
        --per_gpu_train_batch_size=32   \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --output_dir ./tmp/$MODEL/ \
        --logging_steps 500 \
        --evaluate_during_training
        --do_lower_case \ # cased模型不要这一行
    ```
    如果缺少依赖，则pip install相关依赖。


3. 修改paddlenlp相关代码并运行
    ```shell
    cat $HUGGINGFACE/src/transformers/configuration_bert.py | grep \"$MODEL\"
    # wget 上面返回的url地址, 将下载到的json文件中的内容增加到paddlenlp/transformers/model_bert.py 中`BertPreTrainedModel`的`pretrained_init_configuration`内
    cat $HUGGINGFACE/src/transformers/tokenization_bert.py | grep \"$MODEL\"
    # wget 上面返回的vocab的url地址
    # 此外还返回了tokenizer的配置信息，填写到paddlenlp/transformers/tokenizer_bert.py 中`BertTokenizer`的`pretrained_resource_files_map`和`pretrained_init_configuration`内
    ```

    运行代码
   ```shell
   export PDNLP= PaddleNLP repo的本地地址
   export PYTHONPATH=$PYTHONPATH:$PDNLP
   python -u $CONVERT_MODEL/run_glue_pp.py \
        --model_type bert \
        --model_name_or_path $MODEL \
        --task_name QNLI \
        --max_seq_length 128 \
        --batch_size 32   \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --save_steps 500 \
        --output_dir ./tmp/$MODEL/ \
        --n_gpu 1 \
        --params_pd_path params.pd
   ```

   比较输出内容是否与第2步相同，如无问题，模型转换正确。

   将当前目录产生的$MODEL.pdparams模型上传bos，并在paddlenlp/transformers/model_bert.py 中`BertPreTrainedModel`的`pretrained_resource_files_map`中加入该模型对应的链接。

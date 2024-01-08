# 词表扩充预训练文档

词表扩充预训练是自然语言处理（NLP）中的一个重要概念，旨在通过增加模型词汇库的大小和多样性来提高语言模型的理解和生成能力。这个过程通常涉及到识别和添加新的词汇到模型的词表中，使其能够处理更广泛的文本数据。预训练阶段是在模型处理特定任务之前进行的，有助于模型更好地适应不同的语言特征和术语。

## 1.结果展示
词表扩充预训练后，经过sft效果对比表（{<font color=Blue>数字一</font>}/{<font color=Red>数字二</font>}/{<font color=Green>数字三</font>}，分别对应{<font color=Blue>扩充前facebook/llama-7b</font>}/{<font color=Red>扩充后facebook/llama-7b</font>}/{<font color=Green>扩充前的Qwen/qwen-7b</font>}）：


| Dataset | Rouge-1 | Rouge-2 | Rouge-L | BLEU-4 |
| --- | --- | --- | --- | --- |
|WPS|<font color=Blue>0.7895</font>/<font color=Red>**0.8723**</font>/<font color=Green>0.4254</font>|<font color=Blue>0.8374</font>/<font color=Red>**0.8672**</font>/<font color=Green>0.4761</font>|<font color=Blue>0.4661</font>/<font color=Red>**0.9673**</font>/<font color=Green>0.3572</font>|<font color=Blue>0.9080</font>/<font color=Red>**0.9146**</font>/<font color=Green>0.5687</font>|
|HCG|<font color=Blue>0.4962</font>/<font color=Red>**0.6419**</font>/<font color=Green>0.5541</font>|<font color=Blue>0.2444</font>/<font color=Red>**0.4258**</font>/<font color=Green>0.3427</font>|<font color=Blue>0.2601</font>/<font color=Red>**0.3958**</font>/<font color=Green>0.2137</font>|<font color=Blue>0.1939</font>/<font color=Red>**0.3333**</font>/<font color=Green>0.1478</font>|
|GSG|<font color=Blue>0.2088</font>/<font color=Red>**0.6864**</font>/<font color=Green>0.5185</font>|<font color=Blue>0.0304</font>/<font color=Red>**0.5404**</font>/<font color=Green>0.3762</font>|<font color=Blue>0.1330</font>/<font color=Red>**0.5573**</font>/<font color=Green>0.3293</font>|<font color=Blue>0.0095</font>/<font color=Red>**0.5229**</font>/<font color=Green>0.2771</font>|
|BELLE|<font color=Blue>0.4584</font>/<font color=Red>**0.4454**</font>/<font color=Green>0.4738</font>|<font color=Blue>0.2659</font>/<font color=Red>**0.2486**</font>/<font color=Green>0.2666</font>|<font color=Blue>0.4702</font>/<font color=Red>**0.4746**</font>/<font color=Green>0.2785</font>|<font color=Blue>0.2400</font>/<font color=Red>**0.2203**</font>/<font color=Green>0.1768</font>|

结果柱状图：
![avatar](./vocab_extend_results.png)

## 2.快速开始
```shell
# 环境下载
git clone https://github.com/PaddlePaddle/PaddleNLP.git
# pip install ./PaddleNLP 使用develop版本
cd PaddleNLP/llm
# 到达运行目录

cd vocab_extend_scripts/

# 训练新的Tokenizer
python train_tokenizer_model.py --pretrain_files_dir '/path/to/pretrain_files' --model_prefix 'the_name_of_the_tokenizer_model_prefix' --model_type 'the_model_type_used_for_training_tokenizer_model' --vocab_size 'a_number'

# 将训练的新的词表和原始词表进行合并
python merge_tokenizer.py --origin_tokenizer_dir '/path/to/origin_tokenizer/' --chinese_sp_model_file '/path/to/new/tokenizer/' --pretrain_files_dir '/path/to/pretrain_files' --chinese_sp_vocab_file '/path/to/new/*.vocab' --output_dir '/merged/result/output'

#修改待扩充的模型词表参数并保存
python ./vocab_extend_scripts/model_param_vocab_resize.py

# 词表扩充预训练
cd ..
python -u -m paddle.distributed.launch --gpus "0,1,2,3" run_pretrain.py ./vocab_extend_scripts/vocab_extend_json_file/pretrain_argument_vocab_extend.json

#训练结果参数和主干模型合并
python merge_lora_params.py --model_name_or_path  "" --lora_path "" --merge_lora_model_path "" --use_vocab_extend true
```
**Note**: 为了实现TP(Tensor Parallel)分布式策略，需要使用model_param_vocab_resize.py先保存词表扩充后的模型参数，再进行训练

## 3.参数介绍
- `pretrain_files_dir`：
- `model_prefix、model_type、vocab_size`: 新的tokenzer model 名字的前缀：关于sentencePiece的使用参考https://github.com/google/sentencepiece
- `origin_tokenizer_dir`: 扩充前的词表路径
- `chinese_sp_model_file`:新训练的词表路径
- `pretrain_files_dir`:预训练的数据路径，该参数的目的是为了需要将词表设置为8的倍数方便进行TP(Tensor Parallel)策略需要
- `chinese_sp_vocab_file`:新训练的词表路径下的vocab后缀的文件
- `output_dir` 合并后的词表文件路径

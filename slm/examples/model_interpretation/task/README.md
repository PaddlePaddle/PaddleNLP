### 基线模型预测
#### 情感分析：
    预测：model_interpretation/rationale_extraction/sentiment_pred.py
    参数设置参考：model_interpretation/rationale_extraction/run_2_pred_senti_per.sh （参数涉及模型、文件等路径，以及语言的，请根据实际情况进行修改）
#### 文本相似度：
    预测：model_interpretation/rationale_extraction/similarity_pred.py
    参数设置参考：model_interpretation/rationale_extraction/run_2_pred_similarity_per.sh（参数涉及模型、文件等路径，以及语言的，请根据实际情况进行修改）
#### 阅读理解：
    预测：model_interpretation/rationale_extraction/mrc_pred.py
    参数设置参考：model_interpretation/rationale_extraction/run_2_pred_mrc_per.sh（参数涉及模型、文件等路径，以及语言的，请根据实际情况进行修改）
### 三个任务的基线模型训练
#### 情感分析
    RoBERTa：model_interpretation/task/senti/pretrained_models/run_train.sh
    LSTM：model_interpretation/task/senti/rnn/lstm_train.sh
#### 文本相似度
    RoBERTa：model_interpretation/task/similarity/pretrained_models/run_train_pointwise.sh
    LSTM：model_interpretation/task/similarity/simnet/lstm_train.sh
#### 阅读理解
    RoBERTa：model_interpretation/task/mrc/run_train_rc.sh

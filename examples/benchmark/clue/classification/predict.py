import json
from tqdm import tqdm
import argparse 

import paddle
from paddlenlp.datasets import load_dataset
import paddlenlp as ppnlp
from model import WSCModel,NLIModel,KeywordRecognitionModel,NLIModel
from model import LongTextClassification,ShortTextClassification,PointwiseMatching
from data import do_wsc_predict,do_predict,do_csl_predict
from data import do_tnews_predict


pred_dict={
    "cluewsc2020": do_wsc_predict,
    "ocnli" : do_predict,
    "csl" : do_csl_predict,
    "cmnli" : do_predict,
    "iflytek" : do_predict,
    "tnews" : do_tnews_predict,
    "afqmc" : do_predict
}

output_json_dict={
    "cluewsc2020":"cluewsc10_predict.json",
    "ocnli" : "ocnli_50k_predict.json",
    "csl" : "csl_predict.json",
    "cmnli" : "cmnli_predict.json",
    "iflytek" : "iflytek_predict.json",
    "tnews" : "tnews10_predict.json",
    "afqmc" : "iflytek_predict.json"
}

model_dict={
    "cluewsc2020":WSCModel,
    "ocnli":NLIModel,
    "csl":KeywordRecognitionModel,
    "cmnli":NLIModel,
    "iflytek":LongTextClassification,
    "tnews":ShortTextClassification,
    "afqmc": PointwiseMatching
}

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to clue")

    parser.add_argument(
        "--model_path",
        default="best_clue_model",
        type=str,
        help="The  path of the checkpoints .",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    print(args.task_name)
    if(args.task_name in ["tnews","cluewsc2020"]):
        train_ds,test_ds = load_dataset('clue', args.task_name, splits=['train','test1.0'])
    else:
        train_ds,test_ds = load_dataset('clue', args.task_name, splits=['train','test'])

    # 使用 ERNIE-Gram 预训练模型
    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
    model=WSCModel(pretrained_model,len(train_ds.label_list))
    # 加载训练好的模型
    state_dict = paddle.load(args.model_path)
    model.load_dict(state_dict)
    
    predict_label=[]
    for i in tqdm(range(len(test_ds))):
        example=test_ds[i]
        label_pred=pred_dict[args.task_name](model,tokenizer,example)
        predict_label.append(label_pred)

        output_submit_file = output_json_dict[args.task_name]
    label_map = {i: label for i, label in enumerate(train_ds.label_list)}
    # 保存标签结果
    with open(output_submit_file, "w") as writer:
        for i, pred in enumerate(predict_label):
            json_d = {}
            json_d['id'] = i
            json_d['label'] = str(label_map[pred])
        writer.write(json.dumps(json_d) + '\n')
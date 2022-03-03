import json
import numpy as np
from tqdm import tqdm
import os
import pickle
import logging
import time
import random
import pandas as pd
import argparse

import paddle
from paddle.io import TensorDataset
import paddlenlp as ppnlp
from paddlenlp.transformers import ErnieForMultipleChoice,ErnieTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from C3_preprocess import c3Processor,convert_examples_to_features


def process_test_data(data_dir,processor, tokenizer,n_class,max_seq_length):
    
    label_list = processor.get_labels()
    test_examples = processor.get_test_examples()
    feature_dir = os.path.join(data_dir,
                               'test_features{}.pkl'.format(max_seq_length))
    if os.path.exists(feature_dir):
        test_features = pickle.load(open(feature_dir, 'rb'))
    else:
        test_features = convert_examples_to_features(test_examples, label_list,
                                                     max_seq_length, tokenizer)
        with open(feature_dir, 'wb') as w:
            pickle.dump(test_features, w)

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []

    for f in test_features:
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        for i in range(n_class):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].input_mask)
            segment_ids[-1].append(f[i].segment_ids)
        label_id.append(f[0].label_id)

    all_input_ids = paddle.to_tensor(input_ids, dtype='int64')
    all_input_mask = paddle.to_tensor(input_mask, dtype='int64')
    all_segment_ids = paddle.to_tensor(segment_ids, dtype='int64')
    all_label_ids = paddle.to_tensor(label_id, dtype='int64')

    test_data = TensorDataset(
        [all_input_ids, all_input_mask, all_segment_ids, all_label_ids])

    return test_data

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model_path",
        default="checkpoints",
        type=str,
        help="The  path of the checkpoints .",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args.model_path)

    test_batch_size = 4
    max_seq_length=512
    max_num_choices=4
    batch_size=4
    output_dir='checkpoints'

    data_dir='data'
    processor = c3Processor(data_dir)

    # MODEL_NAME = "bert-base-chinese"
    MODEL_NAME =args.model_path
    tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)
    # MODEL_NAME = 'checkpoint'
    model = ErnieForMultipleChoice.from_pretrained(MODEL_NAME,
                                                    num_choices=max_num_choices)

    test_data = process_test_data(output_dir,processor, tokenizer,max_num_choices,max_seq_length)

    test_dataloader = paddle.io.DataLoader(dataset=test_data,
                                                batch_size=test_batch_size,
                                                drop_last=True,
                                                num_workers=0)


    

    logits_all = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader):
        with paddle.no_grad():
            logits = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask)
            logits = logits.numpy()
            for i in range(len(logits)):
                logits_all += [logits[i]]

    submission_test = os.path.join(output_dir, "submission_test.json")
    test_preds = [int(np.argmax(logits_)) for logits_ in logits_all]
    with open(submission_test, "w") as f:
        json.dump(test_preds, f)

    data=json.load(open(submission_test))

    print(len(data))

    with open("data/test1.0.json","r",encoding="utf8") as f:
        test_data = json.load(f)

    ids=[]
    for item in test_data:
        # print(item)
        for sub_item in item[1]:
            idx=sub_item['id']
            # print(idx)
            ids.append(idx)
    print(len(ids))
    with open('c310_predict.json','w') as f:
        for idx,item in zip(ids,data):
            f.write('{'+'"id":{},"label":{}'.format(idx,item)+'}\n')
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
import os
import math
from tqdm import tqdm

import paddle
from paddlenlp.data import Pad
from paddlenlp.transformers import T5ForConditionalGeneration

from uie.evaluation.sel2record import (RecordSchema, MapConfig, SEL2Record)
from uie.seq2struct.t5_bert_tokenizer import T5BertTokenizer

special_to_remove = {'<pad>', '</s>'}


def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name, encoding='utf8')]


def schema_to_ssi(schema: RecordSchema):
    # Convert Schema to SSI
    # <spot> spot type ... <asoc> asoc type <text>
    ssi = "<spot>" + "<spot>".join(sorted(schema.type_list))
    ssi += "<asoc>" + "<asoc>".join(sorted(schema.role_list))
    ssi += "<extra_id_2>"
    return ssi


def post_processing(x):
    for special in special_to_remove:
        x = x.replace(special, '')
    return x.strip()


class Predictor:

    def __init__(self,
                 model_path,
                 max_source_length=256,
                 max_target_length=192) -> None:
        self.tokenizer = T5BertTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    @paddle.no_grad()
    def predict(self, text, schema):

        def to_tensor(x):
            return paddle.to_tensor(x, dtype='int64')

        ssi = schema_to_ssi(schema=schema)

        text = [ssi + x for x in text]

        inputs = self.tokenizer(text,
                                return_token_type_ids=False,
                                return_attention_mask=True,
                                max_seq_len=self.max_source_length)

        inputs = {
            'input_ids':
            to_tensor(
                Pad(pad_val=self.tokenizer.pad_token_id)(inputs['input_ids'])),
            'attention_mask':
            to_tensor(Pad(pad_val=0)(inputs['attention_mask'])),
        }

        pred, _ = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self.max_target_length,
        )

        pred = self.tokenizer.batch_decode(pred.numpy())

        return [post_processing(x) for x in pred]


def find_to_predict_folder(folder_name):
    for root, dirs, _ in os.walk(folder_name):
        for dirname in dirs:
            data_name = os.path.join(root, dirname)
            if os.path.exists(os.path.join(data_name, 'record.schema')):
                yield data_name


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        '-d',
                        required=True,
                        help='Folder need to been predicted.')
    parser.add_argument('--model',
                        '-m',
                        required=True,
                        help='Trained model for inference')
    parser.add_argument('--max_source_length',
                        default=384,
                        type=int,
                        help='Max source length for inference, ssi + text')
    parser.add_argument('--max_target_length', default=192, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument(
        '-c',
        '--config',
        dest='map_config',
        help='Offset mapping config, maping generated sel to offset record',
        default='longer_first_zh')
    parser.add_argument('--verbose', action='store_true')
    options = parser.parse_args()

    # Find the folder need to be predicted with `record.schema`
    data_folder = find_to_predict_folder(options.data)
    model_path = options.model

    predictor = Predictor(model_path=model_path,
                          max_source_length=options.max_source_length,
                          max_target_length=options.max_target_length)

    for task_folder in data_folder:

        print(f'Extracting on {task_folder}')
        schema = RecordSchema.read_from_file(
            os.path.join(task_folder, 'record.schema'))
        sel2record = SEL2Record(
            schema_dict=SEL2Record.load_schema_dict(task_folder),
            map_config=MapConfig.load_by_name(options.map_config),
            tokenizer=predictor.tokenizer,
        )

        test_filename = os.path.join(f"{task_folder}", "test.json")
        if not os.path.exists(test_filename):
            print(f'{test_filename} not found, skip ...')
            continue

        instances = read_json_file(test_filename)
        text_list = [x['text'] for x in instances]
        token_list = [list(x['text']) for x in instances]

        batch_num = math.ceil(len(text_list) / options.batch_size)

        predict = list()
        for index in tqdm(range(batch_num)):
            start = index * options.batch_size
            end = index * options.batch_size + options.batch_size
            predict += predictor.predict(text_list[start:end], schema=schema)

        records = list()
        for p, text, tokens in zip(predict, text_list, token_list):
            records += [sel2record.sel2record(pred=p, text=text, tokens=tokens)]

        pred_filename = os.path.join(f"{task_folder}", "pred.json")
        with open(pred_filename, 'w', encoding='utf8') as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()

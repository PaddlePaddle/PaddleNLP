#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
import os
import re
import math
from tqdm import tqdm

import paddle
from paddlenlp.transformers import T5ForConditionalGeneration

from uie.evaluation.sel2record import (
    RecordSchema,
    MapConfig,
    SEL2Record
)
from uie.seq2struct.t5_bert_tokenizer import T5BertTokenizer


special_to_remove = {'<pad>', '</s>'}


def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name)]


def schema_to_prompt(schema: RecordSchema):
    prompt = "<spot> " + "<spot> ".join(sorted(schema.type_list))
    prompt += "<asoc> " + "<asoc> ".join(sorted(schema.role_list))
    prompt += "<extra_id_2> "
    return prompt


def post_processing(x):
    for special in special_to_remove:
        x = x.replace(special, '')
    return x.strip()


class PaddlePredictor:
    def __init__(self, model_path, max_source_length=256, max_target_length=192) -> None:
        self._tokenizer = T5BertTokenizer.from_pretrained(model_path)
        self._model = T5ForConditionalGeneration.from_pretrained(model_path)
        self._model.eval()
        self._max_source_length = max_source_length
        self._max_target_length = max_target_length

    def get_tokenizer(self):
        return self._tokenizer

    @paddle.no_grad()
    def predict(self, text, schema):
        prompt_ids = self._tokenizer(
            schema_to_prompt(schema),
            return_token_type_ids=False,
            return_attention_mask=False
        )['input_ids'][:-1]

        def process(x):
            return self._tokenizer(
                x,
                return_token_type_ids=False,
                return_attention_mask=False,
                max_seq_len=self._max_source_length
            )

        def process_tensor(x):
            return paddle.to_tensor(x, dtype='int64')[:, :self._max_source_length]

        raw_inputs = [{
            'input_ids': prompt_ids + process(x)['input_ids']} for x in text
        ]

        max_length = max([len(x['input_ids']) for x in raw_inputs])
        inputs = {
            'input_ids': [x['input_ids'] + [self._tokenizer.pad_token_id] * (max_length - len(x['input_ids'])) for x in raw_inputs],
            'attention_mask': [[1] * len(x['input_ids']) + [0] * (max_length - len(x['input_ids'])) for x in raw_inputs],
        }

        inputs['input_ids'] = process_tensor(inputs['input_ids'])
        inputs['attention_mask'] = process_tensor(inputs['attention_mask'])

        result = self._model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self._max_target_length,
        )
        pred, _ = result
        return [self._tokenizer.decode(x, clean_up_tokenization_spaces=False) for x in pred.numpy()]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=True, nargs='+')
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--max_source_length', default=384, type=int)
    parser.add_argument('--max_target_length', default=192, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('-c', '--config', dest='map_config',
                        help='Offset Re-mapping Config',
                        default='longer_first_zh')
    parser.add_argument('--verbose', action='store_true')
    options = parser.parse_args()

    data_folder = options.data
    model_path = options.model

    predictor = PaddlePredictor(
        model_path=model_path,
        max_source_length=options.max_source_length,
        max_target_length=options.max_target_length
    )

    for task_folder in data_folder:
    
        print(f'Extracting on {task_folder}')
        schema = RecordSchema.read_from_file(os.path.join(task_folder, 'record.schema'))
        sel2record = SEL2Record(
            schema_dict=SEL2Record.load_schema_dict(task_folder),
            map_config=MapConfig.load_by_name(options.map_config),
            tokenizer=predictor.get_tokenizer(),
        )

        test_filename = os.path.join(f"{task_folder}", "test.json")
        if not os.path.exists(test_filename):
            print(f'{test_filename} not found, skip ...')
            continue

        text_list = [x['text'] for x in read_json_file(test_filename)]
        char_list = [list(text) for text in text_list]

        batch_num = math.ceil(len(text_list) / options.batch_size)

        predict = list()
        for index in tqdm(range(batch_num)):
            start = index * options.batch_size
            end = index * options.batch_size + options.batch_size

            pred_seq2seq = predictor.predict(text_list[start: end], schema=schema)
            predict += [post_processing(x) for x in pred_seq2seq]

        records = list()
        for p, text, tokens in zip(predict, text_list, char_list):
            r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
            records += [r]
        
        pred_filename = os.path.join(f"{task_folder}", "pred.json")
        with open(pred_filename, 'w') as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()

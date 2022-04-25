#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import re
import math
from tqdm import tqdm

import paddle
from paddlenlp.transformers import (T5Tokenizer, T5ForConditionalGeneration)

from uie.extraction.record_schema import RecordSchema
from uie.sel2record.record import MapConfig
from uie.extraction.scorer import (EntityScorer, RelationScorer, EventScorer)
from uie.sel2record.sel2record import SEL2Record
from uie.seq2seq.t5_bert_tokenizer import T5BertTokenizer

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
    def __init__(self,
                 model_path,
                 schema_file,
                 max_source_length=256,
                 max_target_length=192,
                 use_char_bert=False) -> None:
        if use_char_bert:
            self._tokenizer = T5BertTokenizer.from_pretrained(model_path)
        else:
            self._tokenizer = T5Tokenizer.from_pretrained(
                model_path, keep_accents=True)
        self._model = T5ForConditionalGeneration.from_pretrained(model_path)
        self._model.eval()
        self._schema = RecordSchema.read_from_file(schema_file)
        self._prompt_ids = self._tokenizer(
            schema_to_prompt(self._schema),
            return_token_type_ids=False,
            return_attention_mask=False)['input_ids'][:-1]
        self._max_source_length = max_source_length
        self._max_target_length = max_target_length

    @paddle.no_grad()
    def predict(self, text):
        def process(x):
            return self._tokenizer(
                x,
                return_token_type_ids=False,
                return_attention_mask=False,
                max_seq_len=self._max_source_length)

        raw_inputs = [{
            'input_ids': self._prompt_ids + process(x)['input_ids']
        } for x in text]

        max_length = max([len(x['input_ids']) for x in raw_inputs])
        inputs = {
            'input_ids': [
                x['input_ids'] + [self._tokenizer.pad_token_id] *
                (max_length - len(x['input_ids'])) for x in raw_inputs
            ],
            'attention_mask': [[1] * len(x['input_ids']) + [0] *
                               (max_length - len(x['input_ids']))
                               for x in raw_inputs],
        }

        def process_tensor(x):
            return paddle.to_tensor(
                x, dtype='int64')[:, :self._max_source_length]

        inputs['input_ids'] = process_tensor(inputs['input_ids'])
        inputs['attention_mask'] = process_tensor(inputs['attention_mask'])

        result = self._model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self._max_target_length, )
        pred, prob_score = result
        return [
            self._tokenizer.decode(
                x, clean_up_tokenization_spaces=False) for x in pred.numpy()
        ]


task_dict = {
    'entity': EntityScorer,
    'relation': RelationScorer,
    'event': EventScorer,
}


def eval_performance(records, gold_filename, verbose, match_mode):
    results = dict()
    for task, scorer in task_dict.items():

        gold_list = [x[task] for x in read_json_file(gold_filename)]
        pred_list = [x[task] for x in records]

        gold_instance_list = scorer.load_gold_list(gold_list)
        pred_instance_list = scorer.load_pred_list(pred_list)

        sub_results = scorer.eval_instance_list(
            gold_instance_list=gold_instance_list,
            pred_instance_list=pred_instance_list,
            verbose=verbose,
            match_mode=match_mode, )
        results.update(sub_results)
    print(results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', default='data/text2spotasoc/absa/14lap')
    parser.add_argument(
        '--model', '-m', default='./models/uie_n10_21_50w_absa_14lap')
    parser.add_argument('--max_source_length', default=256, type=int)
    parser.add_argument('--max_target_length', default=192, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument(
        '-c',
        '--config',
        dest='map_config',
        help='Offset Re-mapping Config',
        default='config/offset_map/closest_offset_en.yaml')
    parser.add_argument('--decoding', default='spotasoc')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--match_mode',
        default='normal',
        choices=['set', 'normal', 'multimatch'])
    parser.add_argument('--use_char_bert', action='store_true')
    options = parser.parse_args()

    data_folder = options.data
    model_path = options.model

    predictor = PaddlePredictor(
        model_path=model_path,
        schema_file=f"{data_folder}/record.schema",
        max_source_length=options.max_source_length,
        max_target_length=options.max_target_length,
        use_char_bert=options.use_char_bert)

    map_config = MapConfig.load_from_yaml(options.map_config)
    schema_dict = SEL2Record.load_schema_dict(data_folder)
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=options.decoding,
        map_config=map_config,
        tokenizer=predictor._tokenizer if options.use_char_bert else None, )

    for split, split_name in [('val', 'eval'), ('test', 'test')]:
        gold_filename = f"{data_folder}/{split}.json"

        if not os.path.exists(gold_filename):
            print(f'{gold_filename} not found, skip ...')
            continue

        text_list = [x['text'] for x in read_json_file(gold_filename)]
        token_list = [x['tokens'] for x in read_json_file(gold_filename)]

        batch_num = math.ceil(len(text_list) / options.batch_size)

        predict = list()
        for index in tqdm(range(batch_num)):
            start = index * options.batch_size
            end = index * options.batch_size + options.batch_size

            pred_seq2seq = predictor.predict(text_list[start:end])
            pred_seq2seq = [post_processing(x) for x in pred_seq2seq]

            predict += pred_seq2seq

        records = list()
        for p, text, tokens in zip(predict, text_list, token_list):
            r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
            records += [r]

        eval_performance(
            records=records,
            gold_filename=gold_filename,
            verbose=options.verbose,
            match_mode=options.match_mode, )


if __name__ == "__main__":
    main()

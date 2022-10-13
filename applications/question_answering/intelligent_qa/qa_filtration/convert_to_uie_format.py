import os
import json
from tqdm import tqdm 
import argparse


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source_file_path',
                        type=str,
                        default=None,
                        help='the souce json file path')
    parser.add_argument('--target_file_path',
                        type=str,
                        default=None,
                        help='the target file path')
    parser.add_argument("--tokenizer_path", 
                        type=str, 
                        default="uie-base", 
                        help="The path of tokenizer that you want to load.")
    args = parser.parse_args()
    return args

def convert_from_json_to_uie_format(json_file, output_path, tokenizer=None):
    with open(json_file, 'r', encoding='utf-8') as rf, open(output_path, 'w', encoding='utf-8') as wf:
        for line in rf:
            json_line = json.loads(line)
            context = json_line['context']
            identity = json_line['id']
            answer = json_line['answer']
            if len(answer)>300:
                print(answer)
                answer = answer[:300]
            question = json_line['question']

            prefix = '问题：' + question + '上下文：'
            content = prefix + context

            begin_id = context.find(answer)
            assert begin_id != -1, '\'' + answer + '\' is not found in ' + context 
            end_id = begin_id + len(answer)
            begin_id += len(prefix)
            end_id += len(prefix)

            result = {
                'text': answer, 
                'start': begin_id,
                'end': end_id
            }
            outdict = {
                'content': content, 
                'result_list': [result], 
                'prompt': '答案',
            }
            wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")
                
                    

if __name__ == '__main__':
    args = parse_args()
    convert_from_json_to_uie_format(json_file=args.source_file_path, output_path=args.target_file_path)


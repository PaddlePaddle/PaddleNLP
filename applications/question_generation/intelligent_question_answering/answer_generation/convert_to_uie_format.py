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
    parser.add_argument('--do_answer_prompt',
                        action="store_true",
                        help="is use answer prompt")
    parser.add_argument('--do_len_prompt',
                        action="store_true",
                        help="is use length prompt")
    parser.add_argument('--do_domain_prompt',
                        action="store_true",
                        help="is use domain prompt")
    parser.add_argument('--domain',
                        type=str,
                        default=None,
                        help='the domain of the dataset')
    args = parser.parse_args()
    return args

def convert_from_json_to_uie_format(json_file, output_path, domain=None, do_answer_prompt=True, do_len_prompt=False, do_domain_prompt=False):
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

            begin_id = context.find(answer)
            assert begin_id != -1, '\'' + answer + '\' is not found in ' + context 
            end_id = begin_id + len(answer)
            result = {
                'text': answer, 
                'start': begin_id,
                'end': end_id
            }
            if do_answer_prompt:
                outdict = {
                    'content': context, 
                    'result_list': [result], 
                    'prompt': '答案',
                }
                wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")
            if do_len_prompt:
                if len(answer)<10:
                    len_prompat = '短答案'
                elif len(answer)<20:
                    len_prompat = '中短答案'
                elif len(answer)<30:
                    len_prompat = '中长答案'
                else:
                    len_prompat = '长答案'

                len_outdict = {
                    'content': context, 
                    'result_list': [result], 
                    'prompt': len_prompat,
                }
                wf.write(json.dumps(len_outdict, ensure_ascii=False) + "\n")
            if do_domain_prompt and domain:
                domain_outdict = {
                    'content': context, 
                    'result_list': [result], 
                    'prompt': domain,
                }
                wf.write(json.dumps(domain_outdict, ensure_ascii=False) + "\n")
                
                    


if __name__ == '__main__':
    args = parse_args()
    # convert_from_json_to_uie_format('./raw/DuReaderQG/dev.json', './uie_format/dev.txt')
    # convert_from_json_to_uie_format('./raw/DuReaderQG/train.json', './uie_format/train.txt')
    convert_from_json_to_uie_format(json_file=args.source_file_path, output_path=args.target_file_path, domain=args.domain, do_answer_prompt=args.do_answer_prompt, do_len_prompt=args.do_len_prompt, do_domain_prompt=args.do_domain_prompt)


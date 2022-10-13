import os
import json
from pprint import pprint
from paddlenlp import Taskflow
import re
from tqdm import tqdm

def json_format_indent(json_file, output_json):
    with open(output_json, 'w', encoding='utf-8') as wf:
        with open(json_file, 'r', encoding='utf-8') as rf:
            all_lines = []
            for json_line in rf:
                line_dict = json.loads(json_line)
                all_lines.append(line_dict)
            output_dataset = {'data':all_lines}      
            json.dump(output_dataset, wf, ensure_ascii=False, indent='\t')
        
if __name__ == '__main__':
    json_format_indent('', '')

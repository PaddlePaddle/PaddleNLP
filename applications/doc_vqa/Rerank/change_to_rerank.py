import sys
import json

question = sys.argv[1]

with open('../OCR_process/demo_ocr_res.json', 'r', encoding='utf8') as f:
    paras = []
    for line in f:
        line = json.loads(line.strip())
        document = line['document']
        para = []
        for token in document:
            token = token.replace('▁', '')
            para.append(token)
        paras.append(''.join(para))

with open('./data/demo.tsv', 'w', encoding='utf8') as f:
    for para in paras:
        f.write('{}\t\t{}\t0\n'.format(question, para))

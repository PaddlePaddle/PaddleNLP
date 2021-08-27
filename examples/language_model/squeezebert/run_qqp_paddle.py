import paddle
from paddlenlp.transformers import SqueezeBertModel
from paddlenlp.transformers import SqueezeBertTokenizer
from paddlenlp.transformers import BertModel, BertTokenizer
from utils import *
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--device", default=None, type=str, required=True, )
parser.add_argument("--model_path", default=None, type=str, required=True, )
parser.add_argument("--model_type", default='squeezebert', type=str, required=False, )

args = parser.parse_args()
paddle.set_device(args.device)
model_path = args.model_path

model_class, tokenizer_class = {'bert': [BertModel, BertTokenizer],
                                'squeezebert': [SqueezeBertModel, SqueezeBertTokenizer]}[args.model_type]

tokenizer = tokenizer_class.from_pretrained(model_path)
model = model_class.from_pretrained(model_path)


def read_data():
    import json
    batch_size = 16
    max_len = 128
    res = []
    lines = [json.loads(x) for x in open('./qqp_dev.json', encoding="utf-8")]
    if args.device == 'cpu':
        lines = lines[:1000]
    n_batch = len(lines) // batch_size + 1
    for i in tqdm(range(n_batch)):
        start, end = i * batch_size, min(len(lines), (i + 1) * batch_size)
        data = lines[start: end]
        data = [tokenizer.encode(x['sentence1'], x['sentence2'], max_seq_len=max_len)['input_ids'] for x in data]
        data = sequence_padding(data).astype('int64')
        res.append(data)
    return res


data = read_data()
t = time.time()
model.eval()
for batch in tqdm(data):
    batch = paddle.to_tensor(batch)
    with paddle.no_grad():
        model(batch)
print(time.time() - t)

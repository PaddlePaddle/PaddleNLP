import argparse
import requests
import json

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--prob_limit", default=0.5, type=int, help="probability limit.")

args = parser.parse_args()

url = "http://0.0.0.0:8189/models/text_matching"
headers = {"Content-Type": "application/json"}

if __name__ == "__main__":
    texts = ["三亚是一个美丽的城市","北京烤鸭怎么样"]
    text_pair = ["三亚是个漂亮的城市","北京烤鸭多少钱"]
    
    data = {
        "data": {
            "text": texts,
            "text_pair": text_pair,
        },
        "parameters": {"max_seq_len": args.max_seq_len, 
                        "batch_size": args.batch_size,
                        "prob_limit": args.prob_limit},
    }
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    result_json = json.loads(r.text)
    print(result_json)

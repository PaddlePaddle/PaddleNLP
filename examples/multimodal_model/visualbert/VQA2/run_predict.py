import argparse
import json
import os
import os.path as osp
import re
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddlenlp.data import Dict, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertTokenizer
from paddlenlp.transformers import VisualBertForQuestionAnswering
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--visual_feature_root",
    type=str,
    required=False,
    # default=None,
    default="../X_COCO/data/detectron_fix_100",
    help="Visual feature path.")

parser.add_argument(
    "--mode",
    type=str,
    required=False,
    default="test",
    help="Mode for testing data.")

parser.add_argument(
    "--batch_size", type=int, required=False, default=32, help="Batch images")

parser.add_argument(
    "--model_name_or_path",
    type=str,
    required=False,
    default="visualbert-vqa",
    help="model_name_or_path")

parser.add_argument(
    "--num_classes", type=int, required=False, default=3129, help="num_classes")

parser.add_argument(
    "--return_dict",
    type=bool,
    required=False,
    default=False,
    help="Model return type")

args = parser.parse_args()
# ===================================================================

_num_answers = 10
# ===================================================================

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def tokenize(sentence):
    sentence = sentence.lower()
    sentence = (
        sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s'))
    tokens = SENTENCE_SPLIT_REGEX.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = (self.word2idx_dict['<unk>']
                        if '<unk>' in self.word2idx_dict else None)

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError('word %s not in dictionary \
                             (while dictionary does not contain <unk>)' % w)

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds


# ===================================================================


def compute_answer_scores(answers, num_of_answers, unk_idx):
    scores = np.zeros((num_of_answers), np.float32)
    for answer in set(answers):
        if answer == unk_idx:
            scores[answer] = 0
        else:
            answer_count = answers.count(answer)
            scores[answer] = min(np.float32(answer_count) * 0.3, 1)
    return scores


def generate_test_file(logits, items, answer_dict, out_file):
    assert len(logits) == len(items)
    out_list = []
    for index, item in enumerate(items):
        out_list.append({
            "question_id": item['question_id'],
            "answer": answer_dict.idx2word(logits[index].argmax(0))
        })
    with open(out_file, "w") as f:
        json.dump(out_list, f)


# ===================================================================


def prepare_test_features_single(example, tokenizer, args):
    data_root = args.visual_feature_root

    image_name = example['image_name']
    image_id = example['image_id']
    question_id = example['question_id']
    feature_path = example['feature_path']
    question_str = example['question_str']
    question_tokens = example['question_tokens']

    if args.mode != "test":
        answers = example['answers']
        label = example['label']

    if "train" in feature_path:
        folder = osp.join(data_root, "fc6/vqa/trai n2014")
    elif "val" in feature_path:
        folder = osp.join(data_root, "fc6/vqa/val2014")
    elif "test" in feature_path:
        folder = osp.join(data_root, "fc6/vqa/test2015")

    detectron_features = np.load(os.path.join(folder, feature_path))
    visual_embeds = paddle.to_tensor(detectron_features)
    visual_token_type_ids = paddle.zeros(
        visual_embeds.shape[:-1], dtype=paddle.int64)
    visual_attention_mask = paddle.ones(
        visual_embeds.shape[:-1], dtype=paddle.int64)

    # one sentence for inferencing: (a) Question ? [MASK]
    question_subword_tokens = tokenizer.tokenize(" ".join(question_tokens))
    question_subword_tokens = question_subword_tokens + ["?", "[MASK]"]
    bert_feature = tokenizer.encode(
        question_subword_tokens, return_attention_mask=True)

    data = {
        "input_ids": bert_feature["input_ids"],
        "token_type_ids": bert_feature["token_type_ids"],
        "attention_mask": bert_feature["attention_mask"],
        "question_id": question_id,
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
        "return_dict": False
    }

    if args.mode != "test":
        return data, label

    return data


# ===================================================================

test_ds = load_dataset("vqa2", splits=["test"])
label_list = test_ds.label_list
vocab_info = test_ds.vocab_info

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

test_ds.map(
    partial(
        prepare_test_features_single, tokenizer=tokenizer, args=args),
    batched=False,
    lazy=True,  #!! To save GPU Memory
)

test_batch_sampler = paddle.io.BatchSampler(
    test_ds, batch_size=args.batch_size, shuffle=False)

test_batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "attention_mask": Pad(axis=0, pad_val=0),
    "visual_embeds": Pad(axis=0),
    "visual_token_type_ids": Pad(axis=0),
    "visual_attention_mask": Pad(axis=0), }): fn(samples)

test_data_loader = DataLoader(
    dataset=test_ds,
    batch_sampler=test_batch_sampler,
    collate_fn=test_batchify_fn,
    num_workers=8,
    return_list=True, )

# ===================================================================

answers_vqa_fullname = vocab_info['filepath']
answer_dict = VocabDict(answers_vqa_fullname)

# ===================================================================

model = VisualBertForQuestionAnswering.from_pretrained(
    args.model_name_or_path, num_classes=args.num_classes)
model.eval()

all_logits = []
# ===================================================================

with paddle.no_grad():
    for batch_idx, batch in tqdm(
            enumerate(test_data_loader), total=len(test_data_loader)):
        input_ids, token_type_ids, attention_mask, visual_embeds, visual_token_type_ids, visual_attention_mask = batch
        batch_input = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "return_dict": args.return_dict
        }
        output = model(**batch_input)

        if not args.return_dict:
            logits = output[0]
        else:
            logits = output['logits']
        for idx in range(logits.shape[0]):
            all_logits.append(logits.numpy()[idx])

    generate_test_file(all_logits, test_ds.data, answer_dict, "result.json")

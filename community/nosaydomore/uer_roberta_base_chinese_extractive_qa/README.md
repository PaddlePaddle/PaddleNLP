## uer/roberta-base-chinese-extractive-qa
在中文数据集上训练roberta-base-qa model.
权重来源：https://huggingface.co/uer/roberta-base-chinese-extractive-qa

```python
from paddlenlp.transformers import (
    RobertaModel, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification)
from paddlenlp.transformers import RobertaBPETokenizer, RobertaTokenizer
import paddle
import os
import numpy as np

def decode(start, end, topk, max_answer_len, undesired_tokens):
    """
        Take the output of any :obj:`ModelForQuestionAnswering` and will generate probabilities for each span to be the
        actual answer.
        """
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]
    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    starts, ends = np.unravel_index(idx_sort, candidates.shape)[1:]
    desired_spans = np.isin(starts, undesired_tokens.nonzero()) & np.isin(
        ends, undesired_tokens.nonzero())
    starts = starts[desired_spans]
    ends = ends[desired_spans]
    scores = candidates[0, starts, ends]

    return starts, ends, scores

tokenizer = RobertaTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
questions = ['著名诗歌《假如生活欺骗了你》的作者是']
contexts = [
    '普希金从那里学习人民的语言，吸取了许多有益的养料，这一切对普希金后来的创作产生了很大的影响。这两年里，普希金创作了不少优秀的作品，如《囚徒》、《致大海》、《致凯恩》和《假如生活欺骗了你》等几十首抒情诗，叙事诗《努林伯爵》，历史剧《鲍里斯·戈都诺夫》，以及《叶甫盖尼·奥涅金》前六章。'
]
token = tokenizer(questions, contexts, stride=256, max_seq_len=256)
token_type_ids = token[0]["token_type_ids"]
offset_mapping = token[0]["offset_mapping"]
st_idx = len(token_type_ids) - sum(token_type_ids)

model = RobertaForQuestionAnswering.from_pretrained('uer/roberta-base-chinese-extractive-qa')
model.eval()
input_ids = paddle.to_tensor(
    token[0]['input_ids'], dtype='int64').unsqueeze(0)
token_type_ids = paddle.to_tensor(
    token[0]['token_type_ids'], dtype='int64').unsqueeze(0)
with paddle.no_grad():
    start, end = model(input_ids, token_type_ids)
start_ = start[0].numpy()
end_ = end[0].numpy()
undesired_tokens = np.ones_like(input_ids[0].numpy())

undesired_tokens[1:st_idx] = 0
undesired_tokens[-1] = 0

# Generate mask
undesired_tokens_mask = undesired_tokens == 0.0

# Make sure non-context indexes in the tensor cannot contribute to the softmax
start_ = np.where(undesired_tokens_mask, -10000.0, start_)
end_ = np.where(undesired_tokens_mask, -10000.0, end_)

start_ = np.exp(start_ - np.log(
    np.sum(np.exp(start_), axis=-1, keepdims=True)))
end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))
start_idx, end_idx, score = decode(start_, end_, 1, 64, undesired_tokens)
start_idx, end_idx = offset_mapping[start_idx[0]][0], offset_mapping[
    end_idx[0]][1]
print("ans: {}".format(contexts[0][start_idx:end_idx]),
        'score:{}'.format(score.item()))

'''
ans: 普希金 score:0.9766426086425781
'''

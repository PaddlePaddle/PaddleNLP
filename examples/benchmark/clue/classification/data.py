import numpy as np
import paddle


def convert_example(example, tokenizer):
    encoded_inputs = tokenizer(
        text=example["sentence1"],
        text_pair=example["sentence2"],
        max_seq_len=512,
        pad_to_max_seq_len=True)
    return tuple([
        np.array(
            x, dtype="int64")
        for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
            [example["label"]]
        ]
    ])


def convert_iflytek_example(example, tokenizer):
    encoded_inputs = tokenizer(
        text=example["sentence"], max_seq_len=512, pad_to_max_seq_len=True)
    return tuple([
        np.array(
            x, dtype="int64")
        for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
            [example["label"]]
        ]
    ])


def convert_tnews_example(example, tokenizer):
    encoded_inputs = tokenizer(
        text=example["sentence"], max_seq_len=128, pad_to_max_seq_len=True)
    return tuple([
        np.array(
            x, dtype="int64")
        for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
            [example["label"]]
        ]
    ])


def convert_wsc_example(example, tokenizer):
    text_a = example['text']
    text_a_list = list(text_a)
    target = example['target']
    query = target['span1_text']
    query_idx = target['span1_index']
    pronoun = target['span2_text']
    pronoun_idx = target['span2_index']
    # 给文本的span1_text位置加入"_"标记，然后给文本的span2_text位置加入"[]"
    if pronoun_idx > query_idx:
        text_a_list.insert(query_idx, "_")
        text_a_list.insert(query_idx + len(query) + 1, "_")
        text_a_list.insert(pronoun_idx + 2, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
    else:
        text_a_list.insert(pronoun_idx, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
        text_a_list.insert(query_idx + 2, "_")
        text_a_list.insert(query_idx + len(query) + 2 + 1, "_")

    text_a = "".join(text_a_list)
    encoded_inputs = tokenizer(
        text=text_a, max_seq_len=128, pad_to_max_seq_len=True)
    return tuple([
        np.array(
            x, dtype="int64")
        for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
            [example["label"]]
        ]
    ])


def convert_csl_example(example, tokenizer):
    text_b = example['abst']
    text_a = ';'.join(example['keyword'])
    encoded_inputs = tokenizer(
        text=text_a, text_pair=text_b, max_seq_len=512, pad_to_max_seq_len=True)
    return tuple([
        np.array(
            x, dtype="int64")
        for x in [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
            [example["label"]]
        ]
    ])


def do_wsc_predict(model, tokenizer, example):
    text_a = example['text']
    text_a_list = list(text_a)
    target = example['target']
    query = target['span1_text']
    query_idx = target['span1_index']
    pronoun = target['span2_text']
    pronoun_idx = target['span2_index']
    if pronoun_idx > query_idx:
        text_a_list.insert(query_idx, "_")
        text_a_list.insert(query_idx + len(query) + 1, "_")
        text_a_list.insert(pronoun_idx + 2, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
    else:
        text_a_list.insert(pronoun_idx, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
        text_a_list.insert(query_idx + 2, "_")
        text_a_list.insert(query_idx + len(query) + 2 + 1, "_")

    text_a = "".join(text_a_list)
    # 把文本转换成input_ids,token_type_ids
    encoded_text = tokenizer(
        text=text_a, max_seq_len=128, pad_to_max_seq_len=True)
    # 把input_ids变成paddle tensor
    input_ids = paddle.to_tensor([encoded_text['input_ids']])
    # 把token_type_ids变成paddle tensor
    segment_ids = paddle.to_tensor([encoded_text['token_type_ids']])
    # 模型预测
    pooled_output = model(input_ids, segment_ids)
    # 取概率值最大的索引
    out2 = paddle.argmax(pooled_output, axis=1)
    return out2.numpy()[0]


def do_predict(model, tokenizer, example):
    # 把文本转换成input_ids,token_type_ids
    encoded_text = tokenizer(
        text=example["sentence1"],
        text_pair=example["sentence2"],
        max_seq_len=512,
        pad_to_max_seq_len=True)
    # 把input_ids变成paddle tensor
    input_ids = paddle.to_tensor([encoded_text['input_ids']])
    # 把token_type_ids变成paddle tensor
    segment_ids = paddle.to_tensor([encoded_text['token_type_ids']])
    # 模型预测
    pooled_output = model(input_ids, segment_ids)
    # 取概率值最大的索引
    out2 = paddle.argmax(pooled_output, axis=1)
    return out2.numpy()[0]


def do_csl_predict(model, tokenizer, example):
    # 把文本转换成input_ids,token_type_ids
    text_b = example['abst']
    text_a = ';'.join(example['keyword'])
    encoded_text = tokenizer(
        text=text_a, text_pair=text_b, max_seq_len=512, pad_to_max_seq_len=True)
    # 把input_ids变成paddle tensor
    input_ids = paddle.to_tensor([encoded_text['input_ids']])
    # 把token_type_ids变成paddle tensor
    segment_ids = paddle.to_tensor([encoded_text['token_type_ids']])
    # 模型预测
    pooled_output = model(input_ids, segment_ids)
    # 取概率值最大的索引
    out2 = paddle.argmax(pooled_output, axis=1)
    return out2.numpy()[0]


def do_tnews_predict(model, tokenizer, example):
    # 把文本转换成input_ids,token_type_ids
    encoded_text = tokenizer(
        text=example["sentence"], max_seq_len=512, pad_to_max_seq_len=True)
    # 把input_ids变成paddle tensor
    input_ids = paddle.to_tensor([encoded_text['input_ids']])
    # 把token_type_ids变成paddle tensor
    segment_ids = paddle.to_tensor([encoded_text['token_type_ids']])
    # 模型预测
    pooled_output = model(input_ids, segment_ids)
    # 取概率值最大的索引
    out2 = paddle.argmax(pooled_output, axis=1)
    return out2.numpy()[0]

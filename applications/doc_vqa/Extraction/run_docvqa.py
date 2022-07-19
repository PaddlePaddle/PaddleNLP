import os
import sys
import copy
import json
import random
import logging
import warnings
import argparse
import numpy as np
from collections import OrderedDict, Counter

import paddle
from paddle.static import InputSpec
from paddle.jit import to_static
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer

from docvqa import DocVQA
from model import LayoutXLMForTokenClassification_with_CRF

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--do_train", default=False, type=bool, required=False)
    parser.add_argument("--do_test", default=False, type=bool, required=False)
    parser.add_argument("--test_file", default=None, type=str, required=False)
    parser.add_argument("--train_file", default=None, type=str, required=False)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--max_query_length", default=20, type=int)
    parser.add_argument("--max_doc_length", default=512, type=int)
    parser.add_argument("--max_span_num", default=1, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--eval_steps", type=int, default=10, help="eval every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="the initialized checkpoint")
    parser.add_argument("--save_path", type=str, default=None, help="the initialized checkpoint")
    # yapf: enable
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def get_label_maps():
    labels = ["O", "I-ans", "B-ans", "E-ans"]
    label2id_map = {label: idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "train.log")
        if paddle.distributed.get_rank() == 0 else None,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if paddle.distributed.get_rank() == 0 else logging.WARN,
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    label2id_map, id2label_map = get_label_maps()
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

    # dist mode
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    tokenizer = LayoutXLMTokenizer.from_pretrained(args.model_name_or_path)

    if args.do_test:
        model = LayoutXLMForTokenClassification_with_CRF.from_pretrained(
            args.init_checkpoint)
        evaluate(args,
                 model,
                 tokenizer,
                 label2id_map,
                 id2label_map,
                 pad_token_label_id,
                 global_step=0)
        exit(0)

    if args.init_checkpoint:
        logger.info('Init checkpoint from {}'.format(args.init_checkpoint))
        model = LayoutXLMForTokenClassification_with_CRF.from_pretrained(
            args.init_checkpoint)
    else:
        base_model = LayoutXLMModel.from_pretrained(args.model_name_or_path)
        model = LayoutXLMForTokenClassification_with_CRF(
            base_model, num_classes=len(label2id_map), dropout=None)

    # dist mode
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    train_dataset = DocVQA(args,
                           tokenizer,
                           label2id_map,
                           max_seq_len=args.max_seq_len,
                           max_query_length=args.max_query_length,
                           max_doc_length=args.max_doc_length,
                           max_span_num=args.max_span_num)

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.per_gpu_train_batch_size, shuffle=False)

    args.train_batch_size = args.per_gpu_train_batch_size * max(
        1, paddle.distributed.get_world_size())

    train_dataloader = paddle.io.DataLoader(train_dataset,
                                            batch_sampler=train_sampler,
                                            num_workers=0,
                                            use_shared_memory=True,
                                            collate_fn=None)

    t_total = len(train_dataloader) * args.num_train_epochs
    # build linear decay with warmup lr sch
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=args.learning_rate,
        decay_steps=t_total,
        end_lr=0.0,
        power=1.0)
    if args.warmup_steps > 0:
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            lr_scheduler,
            args.warmup_steps,
            start_lr=0,
            end_lr=args.learning_rate)

    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler,
                                       parameters=model.parameters(),
                                       epsilon=args.adam_epsilon,
                                       weight_decay=args.weight_decay)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed) = %d",
        args.train_batch_size * paddle.distributed.get_world_size(),
    )
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    set_seed(args)
    best_metrics = None
    for epoch_id in range(args.num_train_epochs):
        print('epoch id:{}'.format(epoch_id))
        for step, batch in enumerate(train_dataloader):
            model.train()
            input_ids, input_mask, segment_ids, bboxes, labels = batch
            if input_ids.shape[0] != args.per_gpu_train_batch_size:
                continue
            outputs = model(input_ids=input_ids,
                            bbox=bboxes,
                            attention_mask=input_mask,
                            token_type_ids=segment_ids,
                            labels=labels,
                            is_train=True)
            # model outputs are always tuple in paddlenlp (see doc)
            loss = outputs[0]
            loss = loss.mean()
            if global_step % 50 == 0:
                logger.info(
                    "[epoch {}/{}][iter: {}/{}] lr: {:.5f}, train loss: {:.5f}, "
                    .format(epoch_id, args.num_train_epochs, step,
                            len(train_dataloader), lr_scheduler.get_lr(),
                            loss.numpy()[0]))

            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()  # Update learning rate schedule
            optimizer.clear_grad()
            global_step += 1

            if paddle.distributed.get_rank(
            ) == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir,
                                          "checkpoint-{}".format(global_step))
                os.makedirs(output_dir, exist_ok=True)
                if paddle.distributed.get_rank() == 0:
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(args,
                                os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)


def _tokenize_chinese_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def fast_f1(text1, text2):
    common_char = Counter(text1) & Counter(text2)
    len_seq1 = len(text1)
    len_seq2 = len(text2)
    len_common = sum(common_char.values())
    if len_common == 0:
        return 0.0
    precision = 1.0 * len_common / len_seq2
    recall = 1.0 * len_common / len_seq1
    return (2.0 * precision * recall) / (precision + recall)


def _normalize(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')', u'“', u'”',
        u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',', u'「', u'」', u'（', u'）',
        u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def calc_f1_score(answer, prediction):
    f1_scores = []
    ans_segs = _tokenize_chinese_chars(_normalize(answer))
    prediction_segs = _tokenize_chinese_chars(_normalize(prediction))
    f1 = fast_f1(prediction_segs, ans_segs)
    return f1


def decode(tokenizer, res):
    sep_id = tokenizer._convert_token_to_id("</s>")
    text_res = []
    all_f1 = []
    save_f1 = []
    for i in range(len(res)):
        input_ids, label_ids, predict_ids, bbox = res[i]
        remove_pos = len(' '.join(
            [str(x) for x in input_ids]).split('2 6 ')[0].strip(' ').split(
                ' ')) + 2  # remove the question bbox and sep bbox
        start_pos = input_ids.index(sep_id)
        query_text = []
        for idx in range(1, start_pos):
            input_id = input_ids[idx]
            query_text.append(tokenizer._convert_id_to_token(int(input_id)))

        #label texts and predict texts
        text_label, text_predict = [], []
        label_bbox_index, predict_bbox_index = [], []
        for idx in range(start_pos + 1, len(input_ids)):
            input_id, label_id, predict_id = input_ids[idx], label_ids[
                idx], predict_ids[idx]

            if label_id in [1, 2, 3]:
                text_label.append(tokenizer._convert_id_to_token(int(input_id)))
                label_bbox_index.append(idx - remove_pos + 1)
            if predict_id in [1, 2, 3]:
                text_predict.append(
                    tokenizer._convert_id_to_token(int(input_id)))
                predict_bbox_index.append(idx - remove_pos + 1)
        text_res.append([
            ''.join(query_text), ''.join(text_label), ''.join(text_predict),
            label_bbox_index, predict_bbox_index
        ])

        f1 = calc_f1_score(''.join(text_label), ''.join(text_predict))
        save_f1.append(f1)

        if len(''.join(text_label)) > 10:
            all_f1.append(f1)
    if len(all_f1) > 0:
        print("F1: ", sum(all_f1) / len(all_f1))

    assert len(text_res) == len(save_f1)
    return text_res


def evaluate(args,
             model,
             tokenizer,
             label2id_map,
             id2label_map,
             pad_token_label_id,
             prefix="",
             global_step=0):

    eval_dataset = DocVQA(args,
                          tokenizer,
                          label2id_map,
                          max_seq_len=512,
                          max_query_length=20,
                          max_doc_length=512,
                          max_span_num=1)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(
        1, paddle.distributed.get_world_size())

    eval_dataloader = paddle.io.DataLoader(eval_dataset,
                                           batch_size=args.eval_batch_size,
                                           num_workers=0,
                                           use_shared_memory=True,
                                           collate_fn=None)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    res = []
    for idx, batch in enumerate(eval_dataloader):
        with paddle.no_grad():
            input_ids, input_mask, segment_ids, bboxes, labels = batch

            if input_ids.shape[0] != args.eval_batch_size:
                continue
            outputs = model(input_ids=input_ids,
                            bbox=bboxes,
                            attention_mask=input_mask,
                            token_type_ids=segment_ids,
                            labels=labels,
                            is_train=False)
            labels = labels.numpy()
            crf_decode = outputs[1].numpy()
            bboxes = bboxes.squeeze().numpy()
            input_ids = input_ids.squeeze(axis=1).numpy()

            for index in range(input_ids.shape[0]):
                res.append([
                    list(input_ids[index]),
                    list(labels[index]),
                    list(crf_decode[index]), bboxes[index]
                ])

    origin_inputs = []
    with open(args.test_file, 'r', encoding='utf8') as f:
        for line in f:
            line = json.loads(line.strip())
            origin_inputs.append({
                'img_name': line['img_name'],
                'question': line['question'],
                'bboxes': line['document_bbox'],
                'img_id': line['img_id']
            })

    text_res = decode(tokenizer, res)

    with open(args.save_path, 'w', encoding='utf8') as f:
        for line_res, line_text, line_label in zip(res, text_res,
                                                   origin_inputs):
            line_json = {}
            line_json['img_name'] = line_label['img_name']
            line_json['img_id'] = line_label['img_id']
            line_json['question'] = line_label['question']
            line_json['label_answer'] = line_text[1]
            line_json['predict_answer'] = line_text[2]
            all_boxes = line_res[3]
            label_bbox_index, predict_bbox_index = line_text[3], line_text[4]
            label_bboxes, predict_bboxes = [], []
            for i in range(len(line_label['bboxes'])):
                if i in label_bbox_index:
                    label_bboxes.append(line_label['bboxes'][i])
                if i in predict_bbox_index:
                    predict_bboxes.append(line_label['bboxes'][i])
            line_json['label_bboxes'] = label_bboxes
            line_json['predict_bboxes'] = predict_bboxes
            json.dump(line_json, f, ensure_ascii=False)
            f.write('\n')


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    main(args)

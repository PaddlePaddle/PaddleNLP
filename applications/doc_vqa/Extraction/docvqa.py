import os
import cv2
import sys
import json
import copy
import collections
import numpy as np
from tqdm import tqdm

import paddle
from paddle.io import Dataset

sys.path.insert(0, "../")


class DocVQAExample(object):

    def __init__(self,
                 question,
                 doc_tokens,
                 doc_boxes=[],
                 answer=None,
                 labels=None,
                 image=None):
        self.question = question
        self.doc_tokens = doc_tokens
        self.doc_boxes = doc_boxes
        self.image = image
        self.answer = answer
        self.labels = labels


class DocVQAFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 example_index,
                 input_ids,
                 input_mask,
                 segment_ids,
                 boxes=None,
                 label=None):
        self.example_index = example_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.boxes = boxes
        self.label = label


class DocVQA(Dataset):

    def __init__(self,
                 args,
                 tokenizer,
                 label2id_map,
                 max_seq_len=512,
                 max_query_length=20,
                 max_doc_length=512,
                 max_span_num=1):
        super(DocVQA, self).__init__()
        self.tokenizer = tokenizer
        self.label2id_map = label2id_map
        self.max_seq_len = max_seq_len
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.max_span_num = max_span_num
        self.sample_list = None
        self.args = args

        self.docvqa_inputs = self.docvqa_input()

    def check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context,
                        num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
            best_span_index = span_index

        return cur_span_index == best_span_index

    def convert_examples_to_features(self, examples, tokenizer, label_map,
                                     max_seq_length, max_span_num,
                                     max_doc_length, max_query_length):

        if "[CLS]" in self.tokenizer.get_vocab():
            start_token = "[CLS]"
            end_token = "[SEP]"
        else:
            start_token = "<s>"
            end_token = "</s>"

        features = []
        total = len(examples)
        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question)
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            all_doc_tokens = example.doc_tokens
            all_doc_boxes_tokens = example.doc_boxes

            cls_token_box = [0, 0, 0, 0]
            sep_token_box = [1000, 1000, 1000, 1000]
            pad_token_box = [0, 0, 0, 0]
            ques_token_box = [0, 0, 0, 0]

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += length

            spans_input_ids = []
            spans_input_mask = []
            spans_segment_ids = []
            spans_boxes_tokens = []
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                if doc_span_index == max_span_num:
                    break
                tokens = []
                boxes_tokens = []
                token_is_max_context = {}
                segment_ids = []
                tokens.append(start_token)
                boxes_tokens.append(cls_token_box)
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    boxes_tokens.append(ques_token_box)
                    segment_ids.append(0)
                tokens.append(end_token)
                boxes_tokens.append(sep_token_box)
                segment_ids.append(0)
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    is_max_context = self.check_is_max_context(
                        doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    boxes_tokens.append(all_doc_boxes_tokens[split_token_index])
                    segment_ids.append(0)

                tokens.append(end_token)
                boxes_tokens.append(sep_token_box)
                segment_ids.append(0)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)
                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    boxes_tokens.append(pad_token_box)
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(boxes_tokens) == max_seq_length

                spans_input_ids.append(input_ids)
                spans_input_mask.append(input_mask)
                spans_segment_ids.append(segment_ids)
                spans_boxes_tokens.append(boxes_tokens)

            # Padding
            # padding spans
            # max_span_num: max_seg_num
            # spans_input_ids: the tokens in each segment
            if len(spans_input_ids) > max_span_num:
                spans_input_ids = spans_input_ids[0:max_span_num]
                spans_input_mask = spans_input_mask[0:max_span_num]
                spans_segment_ids = spans_segment_ids[0:max_span_num]
                spans_boxes_tokens = spans_boxes_tokens[0:max_span_num]
            while len(spans_input_ids) < max_span_num:
                tokens = []
                boxes_tokens = []
                segment_ids = []
                tokens.append(start_token)
                boxes_tokens.append(cls_token_box)
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    boxes_tokens.append(ques_token_box)
                    segment_ids.append(0)
                tokens.append(end_token)
                boxes_tokens.append(sep_token_box)
                segment_ids.append(0)
                tokens.append(end_token)
                boxes_tokens.append(sep_token_box)
                segment_ids.append(0)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    boxes_tokens.append(pad_token_box)
                spans_input_ids.append(input_ids)
                spans_input_mask.append(input_mask)
                spans_segment_ids.append(segment_ids)
                spans_boxes_tokens.append(boxes_tokens)

            # padding labels
            labels = example.labels
            sep_id = tokenizer.convert_tokens_to_ids(end_token)
            labels = ["O"] * (spans_input_ids[0].index(sep_id) + 1) + labels
            if len(labels) > 512:
                labels = labels[:512]

            if len(labels) < 512:
                labels += ["O"] * (512 - len(labels))
            assert len(spans_input_ids[0]) == len(labels)

            label_ids = []
            for lid, l in enumerate(labels):
                if l not in label_map:
                    label_ids.append(0)
                else:
                    label_ids.append(label_map[l])

            feature = DocVQAFeatures(
                example_index=example_index,
                input_ids=spans_input_ids,
                input_mask=spans_input_mask,
                segment_ids=spans_segment_ids,
                boxes=spans_boxes_tokens,
                label=label_ids,
            )
            features.append(feature)
        return features

    def create_examples(self, data, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for sample in tqdm(data, total=len(data)):
            question = sample["question"]
            doc_tokens = sample["document"]
            doc_boxes = sample["document_bbox"]
            labels = sample['labels'] if not is_test else []

            x_min, y_min = min(doc_boxes, key=lambda x: x[0])[0], min(
                doc_boxes, key=lambda x: x[2])[2]
            x_max, y_max = max(doc_boxes, key=lambda x: x[1])[1], max(
                doc_boxes, key=lambda x: x[3])[3]
            width = x_max - x_min
            height = y_max - y_min

            if max(width, height) < 1000:
                scale_x = 1
                scale_y = 1
            else:
                scale_x = 1000 / max(width, height)
                scale_y = 1000 / max(width, height)

            scaled_doc_boxes = [[
                round((b[0] - x_min) * scale_x),
                round((b[2] - y_min) * scale_y),
                round((b[1] - x_min) * scale_x),
                round((b[3] - y_min) * scale_y)
            ] for b in doc_boxes]

            for box, oribox in zip(scaled_doc_boxes, doc_boxes):
                if box[0] < 0:
                    print(box, oribox)
                if box[2] - box[0] < 0:
                    print(box, oribox)
                if box[3] - box[1] < 0:
                    print(box, oribox)
                for pos in box:
                    if pos > 1000:
                        print(width, height, box, oribox)

            example = DocVQAExample(question=question,
                                    doc_tokens=doc_tokens,
                                    doc_boxes=scaled_doc_boxes,
                                    labels=labels)
            examples.append(example)
        return examples

    def docvqa_input(self):
        data = []
        if self.args.do_train:
            dataset = self.args.train_file
        elif self.args.do_test:
            dataset = self.args.test_file
        with open(dataset, 'r', encoding='utf8') as f:
            for index, line in enumerate(f):
                data.append(json.loads(line.strip()))

            # read the examples from train/test xlm files
            examples = self.create_examples(data, is_test=self.args.do_test)

        features = self.convert_examples_to_features(
            examples,
            self.tokenizer,
            self.label2id_map,
            max_seq_length=self.max_seq_len,
            max_doc_length=self.max_doc_length,
            max_span_num=self.max_span_num,
            max_query_length=self.max_query_length)

        all_input_ids = paddle.to_tensor([f.input_ids for f in features],
                                         dtype="int64")
        all_input_mask = paddle.to_tensor([f.input_mask for f in features],
                                          dtype="int64")
        all_segment_ids = paddle.to_tensor([f.segment_ids for f in features],
                                           dtype="int64")
        all_bboxes = paddle.to_tensor([f.boxes for f in features],
                                      dtype="int64")
        all_labels = paddle.to_tensor([f.label for f in features],
                                      dtype="int64")
        self.sample_list = [
            np.array(all_input_ids),
            np.array(all_input_mask),
            np.array(all_segment_ids),
            np.array(all_bboxes),
            np.array(all_labels)
        ]

    def __getitem__(self, idx):
        return self.sample_list[0][idx], self.sample_list[1][
            idx], self.sample_list[2][idx], self.sample_list[3][
                idx], self.sample_list[4][idx]

    def __len__(self, ):
        return self.sample_list[0].shape[0]

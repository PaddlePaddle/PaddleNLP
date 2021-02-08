# coding=utf-8
"""This file is adapted from https://github.com/zihangdai/xlnet"""
import io
import six
import sys
import math
import json
import random
import collections
import gc
import numpy as np

sys.path.append('.')
import squad_utils
from data_utils import SEP_ID, CLS_ID, VOCAB_SIZE

import sentencepiece as spm
from prepro_utils import preprocess_text, encode_ids, encode_pieces, printable_text

SPIECE_UNDERLINE = u'▁'

SEG_ID_P = 0
SEG_ID_Q = 1
SEG_ID_CLS = 2
SEG_ID_PAD = 3


class SquadExample(object):
    """A single training/test example for simple sequence classification.
     For examples without an answer, the start and end position are -1.
  """

    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (printable_text(self.qas_id))
        s += ", question_text: %s" % (printable_text(self.question_text))
        s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tok_start_to_orig_index,
                 tok_end_to_orig_index,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 p_mask,
                 segment_ids,
                 paragraph_len,
                 cls_index,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tok_start_to_orig_index = tok_start_to_orig_index
        self.tok_end_to_orig_index = tok_end_to_orig_index
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.p_mask = p_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.cls_index = cls_index
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with io.open(input_file, "r", encoding="utf8") as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                orig_answer_text = None
                is_impossible = False

                if is_training:
                    is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer."
                        )
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        start_position = answer["answer_start"]
                    else:
                        start_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    paragraph_text=paragraph_text,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    is_impossible=is_impossible)
                examples.append(example)

    return examples


def _convert_index(index, pos, M=None, is_start=True):
    if index[pos] is not None:
        return index[pos]
    N = len(index)
    rear = pos
    while rear < N - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if M is not None and index[front] < M - 1:
            if is_start:
                return index[front] + 1
            else:
                return M - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


def convert_examples_to_features(examples, sp_model, max_seq_length, doc_stride,
                                 max_query_length, is_training, uncased):
    """Loads a data file into a list of `InputBatch`s."""

    cnt_pos, cnt_neg = 0, 0
    unique_id = 1000000000
    max_N, max_M = 1024, 1024
    f = np.zeros((max_N, max_M), dtype=np.float32)

    for (example_index, example) in enumerate(examples):

        if example_index % 100 == 0:
            print('Converting {}/{} pos {} neg {}'.format(
                example_index, len(examples), cnt_pos, cnt_neg))

        query_tokens = encode_ids(
            sp_model, preprocess_text(
                example.question_text, lower=uncased))

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        paragraph_text = example.paragraph_text
        para_tokens = encode_pieces(
            sp_model, preprocess_text(
                example.paragraph_text, lower=uncased))

        chartok_to_tok_index = []
        tok_start_to_chartok_index = []
        tok_end_to_chartok_index = []
        char_cnt = 0
        for i, token in enumerate(para_tokens):
            chartok_to_tok_index.extend([i] * len(token))
            tok_start_to_chartok_index.append(char_cnt)
            char_cnt += len(token)
            tok_end_to_chartok_index.append(char_cnt - 1)

        tok_cat_text = ''.join(para_tokens).replace(SPIECE_UNDERLINE, ' ')
        N, M = len(paragraph_text), len(tok_cat_text)

        if N > max_N or M > max_M:
            max_N = max(N, max_N)
            max_M = max(M, max_M)
            f = np.zeros((max_N, max_M), dtype=np.float32)
            gc.collect()

        g = {}

        def _lcs_match(max_dist):
            f.fill(0)
            g.clear()

            ### longest common sub sequence
            # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
            for i in range(N):

                # note(zhiliny):
                # unlike standard LCS, this is specifically optimized for the setting
                # because the mismatch between sentence pieces and original text will
                # be small
                for j in range(i - max_dist, i + max_dist):
                    if j >= M or j < 0: continue

                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]

                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]

                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                    if (preprocess_text(
                            paragraph_text[i], lower=uncased,
                            remove_space=False) == tok_cat_text[j] and
                            f_prev + 1 > f[i, j]):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1

        max_dist = abs(N - M) + 5
        for _ in range(2):
            _lcs_match(max_dist)
            if f[N - 1, M - 1] > 0.8 * N: break
            max_dist *= 2

        orig_to_chartok_index = [None] * N
        chartok_to_orig_index = [None] * M
        i, j = N - 1, M - 1
        while i >= 0 and j >= 0:
            if (i, j) not in g: break
            if g[(i, j)] == 2:
                orig_to_chartok_index[i] = j
                chartok_to_orig_index[j] = i
                i, j = i - 1, j - 1
            elif g[(i, j)] == 1:
                j = j - 1
            else:
                i = i - 1

        if all(v is None
               for v in orig_to_chartok_index) or f[N - 1, M - 1] < 0.8 * N:
            print('MISMATCH DETECTED!')
            continue

        tok_start_to_orig_index = []
        tok_end_to_orig_index = []
        for i in range(len(para_tokens)):
            start_chartok_pos = tok_start_to_chartok_index[i]
            end_chartok_pos = tok_end_to_chartok_index[i]
            start_orig_pos = _convert_index(
                chartok_to_orig_index, start_chartok_pos, N, is_start=True)
            end_orig_pos = _convert_index(
                chartok_to_orig_index, end_chartok_pos, N, is_start=False)

            tok_start_to_orig_index.append(start_orig_pos)
            tok_end_to_orig_index.append(end_orig_pos)

        if not is_training:
            tok_start_position = tok_end_position = None

        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1

        if is_training and not example.is_impossible:
            start_position = example.start_position
            end_position = start_position + len(example.orig_answer_text) - 1

            start_chartok_pos = _convert_index(
                orig_to_chartok_index, start_position, is_start=True)
            tok_start_position = chartok_to_tok_index[start_chartok_pos]

            end_chartok_pos = _convert_index(
                orig_to_chartok_index, end_position, is_start=False)
            tok_end_position = chartok_to_tok_index[end_chartok_pos]
            assert tok_start_position <= tok_end_position

        def _piece_to_id(x):
            if six.PY2 and isinstance(x, unicode):
                x = x.encode('utf-8')
            return sp_model.PieceToId(x)

        all_doc_tokens = list(map(_piece_to_id, para_tokens))

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_is_max_context = {}
            segment_ids = []
            p_mask = []

            cur_tok_start_to_orig_index = []
            cur_tok_end_to_orig_index = []

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i

                cur_tok_start_to_orig_index.append(tok_start_to_orig_index[
                    split_token_index])
                cur_tok_end_to_orig_index.append(tok_end_to_orig_index[
                    split_token_index])

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(SEG_ID_P)
                p_mask.append(0)

            paragraph_len = len(tokens)

            tokens.append(SEP_ID)
            segment_ids.append(SEG_ID_P)
            p_mask.append(1)

            # note(zhiliny): we put P before Q
            # because during pretraining, B is always shorter than A
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(SEG_ID_Q)
                p_mask.append(1)
            tokens.append(SEP_ID)
            segment_ids.append(SEG_ID_Q)
            p_mask.append(1)

            cls_index = len(segment_ids)
            tokens.append(CLS_ID)
            segment_ids.append(SEG_ID_CLS)
            p_mask.append(0)

            input_ids = tokens

            # The mask has 0 for real tokens and 1 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(1)
                segment_ids.append(SEG_ID_PAD)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(p_mask) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    # continue
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    # note(zhiliny): we put P before Q, so doc_offset should be zero.
                    # doc_offset = len(query_tokens) + 2
                    doc_offset = 0
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 0:
                print("*** Example ***")
                print("unique_id: %s" % (unique_id))
                print("example_index: %s" % (example_index))
                print("doc_span_index: %s" % (doc_span_index))
                print("tok_start_to_orig_index: %s" %
                      " ".join([str(x) for x in cur_tok_start_to_orig_index]))
                print("tok_end_to_orig_index: %s" %
                      " ".join([str(x) for x in cur_tok_end_to_orig_index]))
                print("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y)
                    for (x, y) in six.iteritems(token_is_max_context)
                ]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print("segment_ids: %s" %
                      " ".join([str(x) for x in segment_ids]))

                if is_training and span_is_impossible:
                    print("impossible example span")

                if is_training and not span_is_impossible:
                    pieces = [
                        sp_model.IdToPiece(token)
                        for token in tokens[start_position:(end_position + 1)]
                    ]
                    answer_text = sp_model.DecodePieces(pieces)
                    print("start_position: %d" % (start_position))
                    print("end_position: %d" % (end_position))
                    print("answer: %s" % (printable_text(answer_text)))

                    # note(zhiliny): With multi processing,
                    # the example_index is actually the index within the current process
                    # therefore we use example_index=None to avoid being used in the future.
                    # The current code does not use example_index of training data.
            if is_training:
                feat_example_index = None
            else:
                feat_example_index = example_index

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=feat_example_index,
                doc_span_index=doc_span_index,
                tok_start_to_orig_index=cur_tok_start_to_orig_index,
                tok_end_to_orig_index=cur_tok_end_to_orig_index,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                p_mask=p_mask,
                segment_ids=segment_ids,
                paragraph_len=paragraph_len,
                cls_index=cls_index,
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible)

            unique_id += 1
            if span_is_impossible:
                cnt_neg += 1
            else:
                cnt_pos += 1

            yield feature

    print("Total number of instances: {} = pos {} neg {}".format(
        cnt_pos + cnt_neg, cnt_pos, cnt_neg))


def _check_is_max_context(doc_spans, cur_span_index, position):
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


class DataProcessor(object):
    def __init__(self, spiece_model_file, uncased, max_seq_length, doc_stride,
                 max_query_length):
        self._sp_model = spm.SentencePieceProcessor()
        self._sp_model.Load(spiece_model_file)
        self._uncased = uncased
        self._max_seq_length = max_seq_length
        self._doc_stride = doc_stride
        self._max_query_length = max_query_length

        self.current_train_example = -1
        self.num_train_examples = -1
        self.current_train_epoch = -1

        self.train_examples = None
        self.predict_examples = None
        self.num_examples = {'train': -1, 'predict': -1}

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_train_example, self.current_train_epoch

    def get_examples(self, data_path, is_training):
        examples = read_squad_examples(
            input_file=data_path, is_training=is_training)
        return examples

    def get_num_examples(self, phase):
        if phase not in ['train', 'predict']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'predict'].")
        return self.num_examples[phase]

    def get_features(self, examples, is_training):
        features = convert_examples_to_features(
            examples=examples,
            sp_model=self._sp_model,
            max_seq_length=self._max_seq_length,
            doc_stride=self._doc_stride,
            max_query_length=self._max_query_length,
            is_training=is_training,
            uncased=self._uncased)
        return features

    def data_generator(self,
                       data_path,
                       batch_size,
                       phase='train',
                       shuffle=False,
                       dev_count=1,
                       epoch=1):
        if phase == 'train':
            self.train_examples = self.get_examples(data_path, is_training=True)
            examples = self.train_examples
            self.num_examples['train'] = len(self.train_examples)
        elif phase == 'predict':
            self.predict_examples = self.get_examples(
                data_path, is_training=False)
            examples = self.predict_examples
            self.num_examples['predict'] = len(self.predict_examples)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'predict'].")

        def batch_reader(features, batch_size):
            batch = []
            for (index, feature) in enumerate(features):
                if phase == 'train':
                    self.current_train_example = index + 1
                labels = [feature.unique_id
                          ] if feature.start_position is None else [
                              feature.start_position, feature.end_position,
                              feature.is_impossible
                          ]
                example = [
                    feature.input_ids, feature.segment_ids, feature.input_mask,
                    feature.cls_index, feature.p_mask
                ] + labels

                to_append = len(batch) < batch_size
                if to_append:
                    batch.append(example)
                else:
                    yield batch
                    batch = [example]

            if len(batch) > 0:
                yield batch

        def prepare_batch_data(insts):
            """Generate numpy tensors"""
            input_ids = np.expand_dims(
                np.array([inst[0] for inst in insts]).astype('int64'), axis=-1)
            segment_ids = np.array([inst[1] for inst in insts]).astype('int64')
            input_mask = np.array([inst[2] for inst in insts]).astype('float32')
            cls_index = np.expand_dims(
                np.array([inst[3] for inst in insts]).astype('int64'), axis=-1)
            p_mask = np.array([inst[4] for inst in insts]).astype('float32')

            ret_list = [input_ids, segment_ids, input_mask, cls_index, p_mask]
            if phase == 'train':
                start_positions = np.expand_dims(
                    np.array([inst[5] for inst in insts]).astype('int64'),
                    axis=-1)
                end_positions = np.expand_dims(
                    np.array([inst[6] for inst in insts]).astype('int64'),
                    axis=-1)
                is_impossible = np.expand_dims(
                    np.array([inst[7] for inst in insts]).astype('float32'),
                    axis=-1)
                ret_list += [start_positions, end_positions, is_impossible]
            else:
                unique_ids = np.expand_dims(
                    np.array([inst[5] for inst in insts]).astype('int64'),
                    axis=-1)
                ret_list += [unique_ids]

            return ret_list

        def wrapper():
            for epoch_index in range(epoch):
                if shuffle:
                    random.shuffle(examples)
                if phase == 'train':
                    self.current_train_epoch = epoch_index
                    features = self.get_features(examples, is_training=True)
                else:
                    features = self.get_features(examples, is_training=False)

                all_dev_batches = []
                for batch_insts in batch_reader(features, batch_size):
                    batch_data = prepare_batch_data(batch_insts)
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)

                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []

        return wrapper


_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction", [
        "feature_index", "start_index", "end_index", "start_log_prob",
        "end_log_prob"
    ])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, orig_data,
                      args):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))
    # tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(args.start_n_top):
                for j in range(args.end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * args.end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_start_to_orig_index = feature.tok_start_to_orig_index
            tok_end_to_orig_index = feature.tok_end_to_orig_index
            start_orig_pos = tok_start_to_orig_index[pred.start_index]
            end_orig_pos = tok_end_to_orig_index[pred.end_index]

            paragraph_text = example.paragraph_text
            final_text = paragraph_text[start_orig_pos:end_orig_pos + 1].strip()

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="", start_log_prob=-1e6, end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with io.open(output_prediction_file, "w", encoding="utf8") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + u"\n")

    with io.open(output_nbest_file, "w", encoding="utf8") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + u"\n")

    with io.open(output_null_log_odds_file, "w", encoding="utf8") as writer:
        writer.write(json.dumps(scores_diff_json, indent=4) + u"\n")

    qid_to_has_ans = squad_utils.make_qid_to_has_ans(orig_data)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = squad_utils.get_raw_scores(orig_data, all_predictions)
    out_eval = {}

    squad_utils.find_all_best_thresh_v2(out_eval, all_predictions, exact_raw,
                                        f1_raw, scores_diff_json,
                                        qid_to_has_ans)

    return out_eval


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


if __name__ == '__main__':
    processor = DataProcessor(
        spiece_model_file="xlnet_cased_L-24_H-1024_A-16/spiece.model",
        uncased=False,
        max_seq_length=512,
        doc_stride=128,
        max_query_length=64)

    train_data_generator = processor.data_generator(
        data_path="squad_v2.0/dev-v2.0.json",
        batch_size=32,
        phase='predict',
        shuffle=True,
        dev_count=1,
        epoch=1)

    for (index, sample) in enumerate(train_data_generator()):
        if index < 10:
            print("index:", index)
            for tensor in sample:
                print(tensor.shape)
        else:
            break
    #for (index, example) in enumerate(train_examples):
    #    if index < 5:
    #        print(example)

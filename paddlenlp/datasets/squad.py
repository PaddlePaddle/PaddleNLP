import copy
import collections
import json
import os
import warnings

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddle.io import Dataset
from paddlenlp.utils.env import DATA_HOME
from paddlenlp.transformers.tokenizer_utils import _is_whitespace, _is_control, convert_to_unicode

__all__ = ['SQuAD', 'DuReaderRobust', 'CMRC', 'DRCD']


class SquadExample(object):
    """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SQuAD(Dataset):
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))

    DEV_DATA_URL_V2 = 'https://paddlenlp.bj.bcebos.com/datasets/squad/dev-v2.0.json'
    TRAIN_DATA_URL_V2 = 'https://paddlenlp.bj.bcebos.com/datasets/squad/train-v2.0.json'

    DEV_DATA_URL_V1 = 'https://paddlenlp.bj.bcebos.com/datasets/squad/dev-v1.1.json'
    TRAIN_DATA_URL_V1 = 'https://paddlenlp.bj.bcebos.com/datasets/squad/train-v1.1.json'

    SPLITS = {
        '1.1': {
            'train': META_INFO(
                os.path.join('v1', 'train-v1.1.json'),
                '981b29407e0affa3b1b156f72073b945'),
            'dev': META_INFO(
                os.path.join('v1', 'dev-v1.1.json'),
                '3e85deb501d4e538b6bc56f786231552')
        },
        '2.0': {
            'train': META_INFO(
                os.path.join('v2', 'train-v2.0.json'),
                '62108c273c268d70893182d5cf8df740'),
            'dev': META_INFO(
                os.path.join('v2', 'dev-v2.0.json'),
                '246adae8b7002f8679c027697b0b7cf8')
        }
    }

    def __init__(self,
                 tokenizer,
                 mode='train',
                 version_2_with_negative=False,
                 root=None,
                 doc_stride=128,
                 max_query_length=64,
                 max_seq_length=512,
                 **kwargs):

        self.version_2_with_negative = version_2_with_negative
        self._get_data(root, mode, **kwargs)
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length

        self._transform_func = None

        if mode == 'train':
            self.is_training = True
        else:
            self.is_training = False

        self._read()

        self.features = self.convert_examples_to_feature(
            self.examples,
            tokenizer=self.tokenizer,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            max_seq_length=self.max_seq_length)

    def _get_data(self, root, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        if self.version_2_with_negative:
            filename, data_hash = self.SPLITS['2.0'][mode]
        else:
            filename, data_hash = self.SPLITS['1.1'][mode]
        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            if mode == 'train':
                if self.version_2_with_negative:
                    fullname = get_path_from_url(
                        self.TRAIN_DATA_URL_V2,
                        os.path.join(default_root, 'v2'))
                else:
                    fullname = get_path_from_url(
                        self.TRAIN_DATA_URL_V1,
                        os.path.join(default_root, 'v1'))
            elif mode == 'dev':
                if self.version_2_with_negative:
                    fullname = get_path_from_url(
                        self.DEV_DATA_URL_V2, os.path.join(default_root, 'v2'))
                else:
                    fullname = get_path_from_url(
                        self.DEV_DATA_URL_V1, os.path.join(default_root, 'v1'))
        self.full_path = fullname

    def convert_examples_to_feature(self, examples, tokenizer, max_seq_length,
                                    doc_stride, max_query_length):
        """Loads a data file into a list of `InputBatch`s."""
        unique_id = 1000000000
        features = []
        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer._tokenize(example.question_text)
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer._tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if self.is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if self.is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position +
                                                         1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position,
                 tok_end_position) = self._improve_answer_span(
                     all_doc_tokens, tok_start_position, tok_end_position,
                     tokenizer, example.orig_answer_text)

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
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[
                        split_token_index]

                    is_max_context = self._check_is_max_context(
                        doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = input_ids + [
                    tokenizer.vocab[tokenizer.pad_token]
                    for _ in range(self.max_seq_length - len(input_ids))
                ]
                segment_ids = segment_ids + [
                    tokenizer.vocab[tokenizer.pad_token]
                    for _ in range(self.max_seq_length - len(segment_ids))
                ]
                input_mask = [1] * len(input_ids)

                start_position = None
                end_position = None
                if self.is_training and not example.is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                if self.is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0

                feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible)

                unique_id += 1
                features.append(feature)
        return features

    def _improve_answer_span(self, doc_tokens, input_start, input_end,
                             tokenizer, orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer._tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
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

    def _read(self):
        with open(self.full_path, "r", encoding="utf8") as reader:
            input_data = json.load(reader)["data"]

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if _is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False

                    if self.is_training:
                        if self.version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer."
                            )
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            try:
                                end_position = char_to_word_offset[
                                    answer_offset + answer_length - 1]
                            except:
                                continue

                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""
                    else:
                        if self.version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        orig_answer_text = []
                        if not is_impossible and 'answers' in qa.keys():
                            answers = qa["answers"]
                            for answer in answers:
                                orig_answer_text.append(answer["text"])
                        else:
                            start_position = -1
                            end_position = -1
                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)

        self.examples = examples

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        if self.is_training:
            return feature.input_ids, feature.segment_ids, feature.unique_id, feature.start_position, feature.end_position
        else:
            return feature.input_ids, feature.segment_ids, feature.unique_id


class DuReaderRobust(SQuAD):
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))

    DATA_URL = 'https://dataset-bj.cdn.bcebos.com/qianyan/dureader_robust-data.tar.gz'

    SPLITS = {
        'train': META_INFO(
            os.path.join('dureader_robust-data', 'train.json'),
            '800a3dcb742f9fdf9b11e0a83433d4be'),
        'dev': META_INFO(
            os.path.join('dureader_robust-data', 'dev.json'),
            'ae73cec081eaa28a735204c4898a2222'),
        'test': META_INFO(
            os.path.join('dureader_robust-data', 'test.json'),
            'e0e8aa5c7b6d11b6fc3935e29fc7746f')
    }

    def __init__(self,
                 tokenizer,
                 mode='train',
                 root=None,
                 doc_stride=128,
                 max_query_length=64,
                 max_seq_length=512,
                 **kwargs):

        super(DuReaderRobust, self).__init__(
            tokenizer=tokenizer,
            mode=mode,
            version_2_with_negative=False,
            root=root,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            max_seq_length=max_seq_length,
            **kwargs)

    def _get_data(self, root, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, 'self.__class__.__name__')

        filename, data_hash = self.SPLITS[mode]

        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))

            get_path_from_url(self.DATA_URL, default_root)

        self.full_path = fullname

    def _read(self):
        with open(self.full_path, "r", encoding="utf8") as reader:
            input_data = json.load(reader)["data"]

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                raw_doc_tokens = self.tokenizer.basic_tokenizer.tokenize(
                    paragraph_text)
                doc_tokens = []
                char_to_word_offset = []
                k = 0
                temp_word = ""
                for c in paragraph_text:
                    if not self.tokenizer.basic_tokenizer.tokenize(c):
                        char_to_word_offset.append(k - 1)
                        continue
                    else:
                        temp_word += c
                        char_to_word_offset.append(k)

                    if temp_word == raw_doc_tokens[k]:
                        doc_tokens.append(temp_word)
                        temp_word = ""
                        k += 1

                assert k == len(raw_doc_tokens)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False

                    if self.is_training:
                        if (len(qa["answers"]) != 1):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer."
                            )

                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        try:
                            end_position = char_to_word_offset[
                                answer_offset + answer_length - 1]
                        except:
                            continue

                    else:
                        orig_answer_text = []
                        if 'answers' in qa.keys():
                            answers = qa["answers"]
                            for answer in answers:
                                orig_answer_text.append(answer["text"])

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)

        self.examples = examples


class CMRC(DuReaderRobust):
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))

    DEV_DATA_URL = 'https://paddlenlp.bj.bcebos.com/datasets/cmrc/cmrc2018_dev.json'
    TRAIN_DATA_URL = 'https://paddlenlp.bj.bcebos.com/datasets/cmrc/cmrc2018_train.json'
    TRIAL_DATA_URL = 'https://paddlenlp.bj.bcebos.com/datasets/cmrc/cmrc2018_trial.json'

    SPLITS = {
        'train': META_INFO(
            os.path.join('cmrc2018_train.json'),
            '7fb714b479c7f40fbb16acabd7af0ede'),
        'dev': META_INFO(
            os.path.join('cmrc2018_dev.json'),
            '853b80709ff2d071f9fce196521b843c'),
        'trial': META_INFO(
            os.path.join('cmrc2018_trial.json'),
            '853b80709ff2d071f9fce196521b843c')
    }

    def _get_data(self, root, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)

        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            if mode == 'train':
                fullname = get_path_from_url(self.TRAIN_DATA_URL, default_root)
            elif mode == 'dev':
                fullname = get_path_from_url(self.DEV_DATA_URL, default_root)
            elif mode == 'trial':
                fullname = get_path_from_url(self.TRIAL_DATA_URL, default_root)
        self.full_path = fullname


class DRCD(DuReaderRobust):
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))

    DEV_DATA_URL = 'https://paddlenlp.bj.bcebos.com/datasets/DRCD/DRCD_dev.json'
    TRAIN_DATA_URL = 'https://paddlenlp.bj.bcebos.com/datasets/DRCD/DRCD_training.json'
    TEST_DATA_URL = 'https://paddlenlp.bj.bcebos.com/datasets/DRCD/DRCD_test.json'

    SPLITS = {
        'train': META_INFO(
            os.path.join('DRCD_dev.json'), '7fb714b479c7f40fbb16acabd7af0ede'),
        'dev': META_INFO(
            os.path.join('DRCD_training.json'),
            '853b80709ff2d071f9fce196521b843c'),
        'test': META_INFO(
            os.path.join('DRCD_test.json'), '853b80709ff2d071f9fce196521b843c')
    }

    def _get_data(self, root, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)

        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            if mode == 'train':
                fullname = get_path_from_url(self.TRAIN_DATA_URL, default_root)
            elif mode == 'dev':
                fullname = get_path_from_url(self.DEV_DATA_URL, default_root)
            elif mode == 'test':
                fullname = get_path_from_url(self.TEST_DATA_URL, default_root)
        self.full_path = fullname

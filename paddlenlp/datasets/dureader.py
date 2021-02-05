import copy
import collections
import json
import os
import warnings

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from paddle.io import Dataset
from .squad import InputFeatures, SQuAD

__all__ = ['DuReader', 'DuReaderYesNo']


class DuReaderExample(object):
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
                 question_type=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.question_type = question_type
        self.is_impossible = False


class DuReader(SQuAD):
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))

    DATA_URL = 'https://dataset-bj.cdn.bcebos.com/dureader/dureader_preprocessed.zip'

    SPLITS = {
        'train': META_INFO(
            os.path.join('preprocessed', 'trainset', 'zhidao.train.json'),
            None),
        'dev': META_INFO(
            os.path.join('preprocessed', 'devset', 'zhidao.dev.json'), None),
        'test': META_INFO(
            os.path.join('preprocessed', 'testset', 'zhidao.test.json'), None)
    }

    def __init__(self,
                 tokenizer,
                 mode='train',
                 root=None,
                 doc_stride=128,
                 max_query_length=64,
                 max_seq_length=512,
                 **kwargs):

        super(DuReader, self).__init__(
            tokenizer=tokenizer,
            mode=mode,
            root=root,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            max_seq_length=max_seq_length,
            **kwargs)

    def _get_data(self, root, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, 'DuReader')

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

            fullname = get_path_from_url(
                self.DATA_URL, os.path.join(default_root, 'preprocessed'))

        self.full_path = fullname

    def _read(self):
        examples = []
        data_lines = []
        with open(self.full_path, "r", encoding="utf8") as reader:
            data_lines += reader.readlines()
        with open(
                self.full_path.replace('zhidao', 'search'), "r",
                encoding="utf8") as reader:
            data_lines += reader.readlines()
        for entry in data_lines:
            source = json.loads(entry.strip())
            start_id = None
            end_id = None
            orig_answer_text = None

            if self.is_training:
                if (len(source['answer_spans']) == 0):
                    continue
                if source['answers'] == []:
                    continue
                if (source['match_scores'][0] < 0.7):
                    continue

                docs_index = source['answer_docs'][0]
                start_id = source['answer_spans'][0][0]
                end_id = source['answer_spans'][0][1] + 1

                try:
                    answer_passage_idx = source['documents'][docs_index][
                        'most_related_para']
                except:
                    continue

                doc_tokens = source['documents'][docs_index][
                    'segmented_paragraphs'][answer_passage_idx]

                if source['fake_answers'][0] != "".join(doc_tokens[start_id:
                                                                   end_id]):
                    continue
                orig_answer_text = source['fake_answers'][0]
                end_id = end_id - 1

            else:
                doc_tokens = []
                for doc in source['documents']:
                    para_infos = []
                    for para_tokens in doc['segmented_paragraphs']:
                        question_tokens = source['segmented_question']
                        common_with_question = collections.Counter(
                            para_tokens) & collections.Counter(question_tokens)
                        correct_preds = sum(common_with_question.values())
                        if correct_preds == 0:
                            recall_wrt_question = 0
                        else:
                            recall_wrt_question = float(correct_preds) / len(
                                question_tokens)
                        para_infos.append((para_tokens, recall_wrt_question,
                                           len(para_tokens)))
                    para_infos.sort(key=lambda x: (-x[1], x[2]))
                    for para_info in para_infos[:1]:
                        doc_tokens += para_info[0]
                if 'answers' in source.keys():
                    orig_answer_text = source['answers']

            example = DuReaderExample(
                qas_id=source['question_id'],
                question_text=source['question'].strip(),
                question_type=source['question_type'],
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_id,
                end_position=end_id)

            examples.append(example)

        self.examples = examples


class DuReaderYesNo(Dataset):
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))

    DATA_URL = 'https://dataset-bj.cdn.bcebos.com/qianyan/dureader_yesno-data.tar.gz'

    SPLITS = {
        'train': META_INFO(
            os.path.join('dureader_yesno-data', 'train.json'),
            'c469a0ef3f975cfd705e3553ddb27cc1'),
        'dev': META_INFO(
            os.path.join('dureader_yesno-data', 'dev.json'),
            'c38544f8b5a7b567492314e3232057b5'),
        'test': META_INFO(
            os.path.join('dureader_yesno-data', 'test.json'),
            '1c7a1a3ea5b8992eeaeea017fdc2d55f')
    }

    def __init__(self, mode='train', root=None, **kwargs):

        self._get_data(root, mode, **kwargs)
        self._transform_func = None

        if mode == 'train':
            self.is_training = True
        else:
            self.is_training = False

        self._read()

    def _get_data(self, root, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, 'DuReader')

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
        data_lines = []
        with open(self.full_path, "r", encoding="utf8") as reader:
            data_lines += reader.readlines()

        examples = []
        for entry in data_lines:
            source = json.loads(entry.strip())
            examples.append([
                source['question'], source['answer'], source['yesno_answer'],
                source['id']
            ])

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def get_labels(self):
        """
        Return labels of the DuReaderYesNo sample.
        """
        return ["Yes", "No", "Depends"]

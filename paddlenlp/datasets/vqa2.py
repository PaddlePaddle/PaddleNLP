import collections
import numpy as np
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['VQA2']

class VQA2(DatasetBuilder):
    URL_ANNO = "https://dl.fbaipublicfiles.com/mmf/data/datasets/vqa2/defaults/annotations/annotations.tar.gz"
    MD5_ANNO = 'bb5cd1e9101f8fa78c55bd9c0ec8eaba'
    URL_EXTRA = "https://dl.fbaipublicfiles.com/mmf/data/datasets/vqa2/defaults/extras.tar.gz"
    MD5_EXTRA = '9baaf3291221fdef49852046a500f731'
    # URL_FEATURE = "https://dl.fbaipublicfiles.com/pythia/features/detectron_fix_100.tar.gz"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    # IMG_FEATURE_META_INFO = collections.namedtuple('META_INFO', ('folder'))

    SPLITS = {
        'train': META_INFO(
            os.path.join('annotations', 'imdb_train2014.npy'), '27b9fc1c2393e1e62839ef65f255c645'),
        'val': META_INFO(
            os.path.join('annotations', 'imdb_val2014.npy'), 'd51bec47b76d1efb8bfeac8f0fd1dabc'),
        'trainval': META_INFO(
            os.path.join('annotations', 'imdb_trainval2014.npy'), 'd51bec47b76d1efb8bfeac8f0fd1dabc'),
        'minival': META_INFO(
            os.path.join('annotations', 'imdb_minival2014.npy'), 'dd982ed6d924724482aee42c72d324ce'),
        'test': META_INFO(
            os.path.join('annotations', 'imdb_test2015.npy'), '5acad6cf0839c01c64e056506ffb18c5'),
    }
    
    VOCAB_INFO = META_INFO(os.path.join('extras', 'vocabs', 'answers_vqa.txt'), '971cad957919ff2cf1f5f1c70bef4d90')

    def _get_data(self, mode, **kwargs):
        # VQA annotations
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL_ANNO, os.path.join(default_root, 'annotations'), self.MD5_ANNO)

        # VQA vocab
        answers_vqa_filename, answers_vqa_data_hash = self.VOCAB_INFO
        answers_vqa_fullname = os.path.join(default_root, answers_vqa_filename)
        if not os.path.exists(answers_vqa_fullname) or (answers_vqa_data_hash and
                                            not md5file(answers_vqa_fullname) == answers_vqa_data_hash):
            get_path_from_url(self.URL_EXTRA, os.path.join(default_root), self.MD5_EXTRA)
        
        return fullname
    
    def _read(self, filename, split):
        """Reads data."""
        items = np.load(filename, allow_pickle = True)[1:]
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)

        for index, iminfo in enumerate(items):
            if not isinstance(iminfo, dict):
                raise TypeError("'iminfo' passed to _read must be a dict")
            
            sample = {}
            if 'image_name' not in iminfo.keys():
                continue
            image_name = iminfo['image_name']
            image_id = iminfo['image_id']
            question_id = iminfo['question_id']
            feature_path = iminfo['feature_path']
            question_str = iminfo['question_str']
            question_tokens = iminfo['question_tokens']
            # ocr_tokens = iminfo['ocr_tokens']

            if split == "test":
                answers = None
            else:
                if 'answers' in iminfo.keys():
                    answers = iminfo['answers']
                elif 'valid_answers' in iminfo.keys():
                    answers = iminfo['valid_answers']
                else:
                    raise NotImplementedError("`answers` not found in annatation file: {}".format(filename))
                
            if answers is not None:
                sample['image_name'] = image_name
                sample['image_id'] = image_id
                sample['question_id'] = question_id
                sample['feature_path'] = feature_path
                sample['question_str'] = question_str
                sample['question_tokens'] = question_tokens
                # sample['ocr_tokens'] = ocr_tokens
                sample['answers'] = answers
                sample['split_name'] = split
            else:
                sample['image_name'] = image_name
                sample['image_id'] = image_id
                sample['question_id'] = question_id
                sample['feature_path'] = feature_path
                sample['question_str'] = question_str
                sample['question_tokens'] = question_tokens
                # sample['ocr_tokens'] = ocr_tokens
                sample['split_name'] = split
                
            yield sample
    
    def get_labels(self):
        return ["LABEL_{}".format(idx) for idx in range(3129)]
    
    def get_vocab(self):
        en_vocab_fullname = os.path.join(DATA_HOME, self.__class__.__name__, self.VOCAB_INFO[0])

        # Construct vocab_info to match the form of the input of `Vocab.load_vocabulary()` function
        vocab_info = {
            'filepath': en_vocab_fullname,
            
        }
        return vocab_info
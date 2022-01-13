import collections
import os
import random

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from pycocotools.coco import COCO

from . import DatasetBuilder

__all__ = ['COCOCaptions']


class COCOCaptions(DatasetBuilder):
    URL_ANNO = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    MD5_ANNO = '0a379cfc70b0e71301e0f377548639bd'
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))

    SPLITS = {
        'train': META_INFO(
            os.path.join('annotations', 'captions_train2014.json'),
            'abfcc336d66eec2e8c453600cc1db939'),
        'val': META_INFO(
            os.path.join('annotations', 'captions_val2014.json'),
            '41a3e90178948b2ba004175080277e35'),
    }

    _two_sentence = True

    def _get_data(self, mode, **kwargs):
        # VQA annotations
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL_ANNO,
                              os.path.join(default_root), self.MD5_ANNO)

        return fullname

    def _read(self, filename, split):
        """Reads data."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)

        coco = COCO(filename)
        items = coco.loadAnns(coco.getAnnIds())

        print("{} of captions in total.".format(len(items)))

        for index, iminfo in enumerate(items):
            if not isinstance(iminfo, dict):
                raise TypeError("'iminfo' passed to _read must be a dict")

            sample = {}

            image_id = iminfo["image_id"]
            caption_a = iminfo["caption"]

            rest_anns = coco.loadAnns([
                i for i in coco.getAnnIds(imgIds=image_id) if i != iminfo['id']
            ])
            if self._two_sentence:
                if random.random() > 0.5:
                    item_b = items[random.randint(0, len(items) - 1)]
                    while item_b["image_id"] == image_id:
                        item_b = items[random.randint(0, len(items) - 1)]
                    flag = False
                else:
                    item_b = rest_anns[random.randint(0, len(rest_anns) - 1)]
                    flag = True

            caption_b = item_b["caption"]

            sample['image_id'] = image_id
            sample["split_name"] = split
            sample['caption_a'] = caption_a
            sample['caption_b'] = caption_b
            sample['is_correct'] = flag

            yield sample

    def get_labels(self):
        pass

    def get_vocab(self):
        pass

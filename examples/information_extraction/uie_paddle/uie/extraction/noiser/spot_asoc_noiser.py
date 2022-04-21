#!/usr/bin/env python
# -*- coding:utf-8 -*-
from uie.extraction import constants
from dataclasses import dataclass
import numpy as np
from uie.extraction.utils import *


@dataclass
class SpotAsocNoiser:
    spot_noise_ratio: float = 0.
    asoc_noise_ratio: float = 0.
    null_span: str = constants.null_span

    def random_insert_spot(self, spot_asoc, spot_label_list=None):
        """随机插入 Spot，类别从 spot_label_list 中自动选择

        Args:
            spot_asoc ([type]): [description]
            spot_label_list ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if spot_label_list is None or len(spot_label_list) == 0:
            return spot_asoc
        random_num = sum(
            np.random.binomial(1, self.spot_noise_ratio, len(spot_asoc)))
        for _ in range(random_num):
            random_position = np.random.randint(low=0, high=len(spot_asoc))
            random_label = np.random.choice(spot_label_list)
            spot_asoc.insert(random_position, {
                "span": self.null_span,
                "label": random_label,
                'asoc': list()
            })
        return spot_asoc

    def random_insert_asoc(self, spot_asoc, asoc_label_list=None):
        """随机插入 Asoc，类别从 asoc_label_list 中自动选择

        Args:
            spot_asoc ([type]): [description]
            asoc_label_list ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if asoc_label_list is None or len(asoc_label_list) == 0:
            return spot_asoc
        # asoc_sum = sum([len(x['asoc']) for x in spot_asoc])
        spot_sum = len(spot_asoc)
        random_num = sum(np.random.binomial(1, self.asoc_noise_ratio, spot_sum))
        for _ in range(random_num):
            random_label = np.random.choice(asoc_label_list)
            spot_position = np.random.randint(low=0, high=len(spot_asoc))
            asoc_position = np.random.randint(
                low=0, high=len(spot_asoc[spot_position]['asoc']) + 1)
            spot_asoc[spot_position]['asoc'].insert(
                asoc_position, (random_label, self.null_span))
        return spot_asoc

    def add_noise(self, spot_asoc, spot_label_list, asoc_label_list):
        spot_asoc = self.random_insert_asoc(
            spot_asoc=spot_asoc,
            asoc_label_list=asoc_label_list, )
        spot_asoc = self.random_insert_spot(
            spot_asoc=spot_asoc,
            spot_label_list=spot_label_list, )
        return spot_asoc


def main():
    from uie.extraction.constants import BaseStructureMarker
    structure_marker = BaseStructureMarker()
    spot_asoc = [{
        "span": "analyzer",
        "label": "generic",
        "asoc": []
    }, {
        "span": "`` Amorph ''",
        "label": "method",
        "asoc": []
    }]

    spot_asoc_noiser = SpotAsocNoiser(
        spot_noise_ratio=0.5,
        asoc_noise_ratio=0.5, )
    spot_asoc_noiser.add_noise(
        spot_asoc=spot_asoc,
        spot_label_list=['A', 'B', 'C'],
        asoc_label_list=['D', 'E', 'F'], )
    target = convert_spot_asoc(
        spot_asoc_instance=spot_asoc, structure_maker=structure_marker)

    target = convert_spot_asoc(
        spot_asoc_instance=spot_asoc, structure_maker=structure_marker)

    replace_map = {
        '<extra_id_0>': ' ( ',
        '<extra_id_1>': ' ) ',
        '<extra_id_5>': ':',
    }
    from nltk.tree import Tree
    for old, new in replace_map.items():
        target = target.replace(old, new)
    print(target)
    Tree.fromstring(target).pretty_print()


if __name__ == "__main__":
    main()

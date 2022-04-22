#!/usr/bin/env python
# -*- coding:utf-8 -*-
from uie.extraction import constants
from dataclasses import dataclass
import numpy as np


@dataclass
class SpotAsocNoiser:
    spot_noise_ratio: float = 0.
    asoc_noise_ratio: float = 0.
    null_span: str = constants.null_span

    def random_insert_spot(self, spot_asoc, spot_label_list=None):
        """ Insert negative spot in random, sample negative spot from spot_label_list
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
        """ Insert negative asoc in random, sample negative asoc from asoc_label_list
        """
        if asoc_label_list is None or len(asoc_label_list) == 0:
            return spot_asoc

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

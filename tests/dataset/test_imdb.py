# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import unittest
import paddle
from paddlenlp.datasets import Imdb
from paddlenlp.utils.log import logger

from common_test import CpuCommonTest


class TestImdbTrainSet(CpuCommonTest):
    def set_config(self):
        self.config['mode'] = 'train'

    def test_training_set(self):
        np.random.seed(102)
        expected_text, expected_label = (
            'This is a strange cerebral surreal esoteric film If there is such a thing as intellectual horror cinema this film is it I started to get scared and wish there was someone else watching it with me and it barely has a plot Im going to have to see this film again multiple times before I feel I really understand it If youre the kind of person who likes My Dinner With Andre and films by Godard or if you do a lot of mindaltering drugs you will probably enjoy this film Wow',
            1)
        expected_len = 25000

        train_ds = Imdb(**self.config)
        self.check_output_equal(len(train_ds), expected_len)
        self.check_output_equal(expected_text, train_ds[0][0])
        self.check_output_equal(expected_label, train_ds[0][1])


class TestImdbTestSet(CpuCommonTest):
    def set_config(self):
        self.config['mode'] = 'test'

    def test_training_set(self):
        np.random.seed(102)
        expected_text, expected_label = (
            'Two hardluck but crafty ladies decide to act like HAVANA WIDOWS by sailing to Cuba to meet  blackmail rich gentlemenbr br This was the sort of ephemeral comic frippery which the studios produced quite effortlessly during the 1930s Well made  highly enjoyable Depression audiences couldnt seem to get enough of these popular funny photo dramasbr br Joan Blondell  Glenda Farrell are perfectly cast as the frantic fasttalking females who will go to great lengths to make a little dishonest dough Although Joan gets both top billing and the romantic scenes both gals are as talented  watchable as they are gorgeousbr br Handsome Lyle Talbot plays Joans persistent suitor but hes given relatively little to do Chubby cherubic Guy Kibbee appears as the girls intended target Whether awakening to find himself in the wrong bed or being chased across the roof of a Cuban hacienda in his long johns he is equally hilarious Behind him comes a rank of character actors  Allen Jenkins Frank McHugh Ruth Donnelly Hobart Cavanaugh Maude Eburne Dewey Robinson  all equally adept at pleasing the toughest crowdbr br Movie mavens will recognize an uncredited James Murray as the suspicious bank teller with the forged check This very talented actor was pulled out of complete obscurity to star in King Vidors THE CROWD 1928 one of the silent eras most prestigious films Hopes were high for a great career but his celebrity faded quickly with sound pictures After a long string of tiny roles  bit parts broke  destitute his life ended in the waters of a New York river in 1936 He was only 35 years oldbr br While never stars of the first rank Joan Blondell 19061979  Glenda Farrell 19041971 enlivened scores of films at Warner Bros throughout the 1930s especially the eight in which they appeared together Whether playing gold diggers or working girls reporters or secretaries these blonde  brassy ladies were very nearly always a match for whatever leading man was lucky enough to share equal billing alongside them With a wisecrack or a glance their characters showed they were ready to take on the world  and any man in it Never as wickedly brazen as Paramounts Mae West you always had the feeling that tough as they were Blondell  Farrell used their toughness to defend vulnerable hearts ready to break over the right guy While many performances from seven decades ago can look campy or contrived today these two lovely ladies are still spirited  sassy',
            1)
        expected_len = 25000

        test_ds = Imdb(**self.config)
        self.check_output_equal(len(test_ds), expected_len)
        self.check_output_equal(expected_text, test_ds[0][0])
        self.check_output_equal(expected_label, test_ds[0][1])


class TestImdbWrongMode(CpuCommonTest):
    def set_config(self):
        # valid mode is 'train' and 'test', wrong mode would raise an error
        self.config['mode'] = 'wrong'

    def test_training_set(self):
        with self.assertRaises(AssertionError):
            Imdb(**self.config)

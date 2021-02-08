#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
SimNet reader
"""

import logging
import numpy as np
import io

import tokenization

class SimNetProcessor(object):
    def __init__(self, args, vocab):
        self.args = args
        # load vocab
        self.vocab = vocab
        self.valid_label = np.array([])
        self.test_label = np.array([])
        if args.tokenizer:
            self.tokenizer = getattr(tokenization, args.tokenizer)()
        else:
            self.tokenizer = None

    def get_reader(self, mode, epoch=0):
        """
        Get Reader
        """

        def reader_with_pairwise():
            """
                Reader with Pairwise
            """
            if mode == "valid":
                with io.open(self.args.valid_data_dir, "r",
                                 encoding="utf8") as file:
                    for line in file:
                        query, title, label = line.strip().split("\t")
                        if len(query) == 0 or len(title) == 0 or len(
                                label) == 0 or not label.isdigit() or int(
                                    label) not in [0, 1]:
                            logging.warning(
                                "line not match format in test file")
                            continue

                        # tokenize
                        if self.tokenizer:
                            query = self.tokenizer.tokenize(query)
                            title = self.tokenizer.tokenize(title)

                        query = [
                            self.vocab[word] for word in query.split(" ")
                            if word in self.vocab
                        ]
                        title = [
                            self.vocab[word] for word in title.split(" ")
                            if word in self.vocab
                        ]
                        if len(query) == 0:
                            query = [0]
                        if len(title) == 0:
                            title = [0]
                        yield [query, title]
            elif mode == "test":
                with io.open(self.args.test_data_dir, "r", encoding="utf8") as file:
                    for line in file:
                        query, title, label = line.strip().split("\t")
                        if len(query) == 0 or len(title) == 0 or len(
                                label) == 0 or not label.isdigit() or int(
                                    label) not in [0, 1]:
                            logging.warning(
                                "line not match format in test file")
                            continue

                        # tokenize
                        if self.tokenizer:
                            query = self.tokenizer.tokenize(query)
                            title = self.tokenizer.tokenize(title)

                        query = [
                            self.vocab[word] for word in query.split(" ")
                            if word in self.vocab
                        ]
                        title = [
                            self.vocab[word] for word in title.split(" ")
                            if word in self.vocab
                        ]
                        if len(query) == 0:
                            query = [0]
                        if len(title) == 0:
                            title = [0]
                        yield [query, title]
            else:
                for idx in range(epoch):
                    with io.open(self.args.train_data_dir, "r",
                                    encoding="utf8") as file:
                        for line in file:
                            query, pos_title, neg_title = line.strip().split("\t")
                            if len(query) == 0 or len(pos_title) == 0 or len(
                                    neg_title) == 0:
                                logging.warning(
                                    "line not match format in test file")
                                continue
                        # tokenize
                        if self.tokenizer:
                            query = self.tokenizer.tokenize(query)
                            pos_title = self.tokenizer.tokenize(pos_title)
                            neg_title = self.tokenizer.tokenize(neg_title)

                            query = [
                                self.vocab[word] for word in query.split(" ")
                                if word in self.vocab
                            ]
                            pos_title = [
                                self.vocab[word] for word in pos_title.split(" ")
                                if word in self.vocab
                            ]
                            neg_title = [
                                self.vocab[word] for word in neg_title.split(" ")
                                if word in self.vocab
                            ]
                            if len(query) == 0:
                                query = [0]
                            if len(pos_title) == 0: 
                                pos_title = [0]
                            if len(neg_title) == 0:
                                neg_title = [0]
                            yield [query, pos_title, neg_title]

        def reader_with_pointwise():
            """
            Reader with Pointwise
            """
            if mode == "valid":
                with io.open(self.args.valid_data_dir, "r",
                                 encoding="utf8") as file:
                    for line in file:
                        query, title, label = line.strip().split("\t")
                        if len(query) == 0 or len(title) == 0 or len(
                                label) == 0 or not label.isdigit() or int(
                                    label) not in [0, 1]:
                            logging.warning(
                                "line not match format in test file")
                            continue

                        # tokenize
                        if self.tokenizer:
                            query = self.tokenizer.tokenize(query)
                            title = self.tokenizer.tokenize(title)

                        query = [
                            self.vocab[word] for word in query.split(" ")
                            if word in self.vocab
                        ]
                        title = [
                            self.vocab[word] for word in title.split(" ")
                            if word in self.vocab
                        ]
                        if len(query) == 0:
                            query = [0]
                        if len(title) == 0:
                            title = [0]
                        yield [query, title]
            elif mode == "test":
                with io.open(self.args.test_data_dir, "r", encoding="utf8") as file:
                    for line in file:
                        query, title, label = line.strip().split("\t")
                        if len(query) == 0 or len(title) == 0 or len(
                                label) == 0 or not label.isdigit() or int(
                                    label) not in [0, 1]:
                            logging.warning(
                                "line not match format in test file")
                            continue

                        # tokenize
                        if self.tokenizer:
                            query = self.tokenizer.tokenize(query)
                            title = self.tokenizer.tokenize(title)

                        query = [
                            self.vocab[word] for word in query.split(" ")
                            if word in self.vocab
                        ]
                        title = [
                            self.vocab[word] for word in title.split(" ")
                            if word in self.vocab
                        ]
                        if len(query) == 0:
                            query = [0]
                        if len(title) == 0:
                            title = [0]
                        yield [query, title]
            else:
                for idx in range(epoch):
                    with io.open(self.args.train_data_dir, "r",
                                    encoding="utf8") as file:
                        for line in file:
                            query, title, label = line.strip().split("\t")
                            if len(query) == 0 or len(title) == 0 or len(
                                    label) == 0 or not label.isdigit() or int(
                                        label) not in [0, 1]:
                                logging.warning(
                                    "line not match format in test file")
                                continue

                            # tokenize
                            if self.tokenizer:
                                query = self.tokenizer.tokenize(query)
                                title = self.tokenizer.tokenize(title)

                            query = [
                                self.vocab[word] for word in query.split(" ")
                                if word in self.vocab
                            ]
                            title = [
                                self.vocab[word] for word in title.split(" ")
                                if word in self.vocab
                            ]
                            label = int(label)
                            if len(query) == 0:
                                query = [0]
                            if len(title) == 0:
                                title = [0]
                            yield [query, title, label]

        if self.args.task_mode == "pairwise":
            return reader_with_pairwise
        else:
            return reader_with_pointwise

    def get_infer_reader(self):
        """
        get infer reader
        """
        with io.open(self.args.infer_data_dir, "r", encoding="utf8") as file:
            for line in file:
                query, title = line.strip().split("\t")
                if len(query) == 0 or len(title) == 0:
                    logging.warning("line not match format in test file")
                    continue
                # tokenize
                if self.tokenizer:
                    query = self.tokenizer.tokenize(query)
                    title = self.tokenizer.tokenize(title)
                query = [
                    self.vocab[word] for word in query.split(" ")
                    if word in self.vocab
                ]
                title = [
                    self.vocab[word] for word in title.split(" ")
                    if word in self.vocab
                ]
                if len(query) == 0:
                    query = [0]
                if len(title) == 0:
                    title = [0]
                yield [query, title]

    def get_infer_data(self):
        """
        get infer data
        """
        with io.open(self.args.infer_data_dir, "r", encoding="utf8") as file:
            for line in file:
                query, title = line.strip().split("\t")
                if len(query) == 0 or len(title) == 0:
                    logging.warning("line not match format in test file")
                    continue
                yield line.strip()

    def get_valid_label(self):
        """
        get valid data label
        """
        if self.valid_label.size == 0:
            labels = []
            with io.open(self.args.valid_data_dir, "r", encoding="utf8") as f:
                for line in f:
                    labels.append([int(line.strip().split("\t")[-1])])
            self.valid_label = np.array(labels)
        return self.valid_label

    def get_test_label(self):
        """
        get test data label
        """
        if self.test_label.size == 0:
            labels = []
            with io.open(self.args.test_data_dir, "r", encoding="utf8") as f:
                for line in f:
                    labels.append([int(line.strip().split("\t")[-1])])
            self.test_label = np.array(labels)
        return self.test_label

#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class CovidDataset(paddle.io.Dataset):
    """
    CovidDataset is used to process the data downloaded from CSSEGISandData.

    Args:
        path (obj:`str`, required): the data path.
        test_data_size (obj:`int`, required): The data will be split to a train set and a test set. test_data_size determines the test set size.
        seq_length (obj:`int`, required): The data will be organized as small time series. seq_length determines each time series length.
        mode (obj:`str`, optional): The load mode, "train", "test" or "infer". Defaults to 'train', meaning load the train dataset.
    """

    def __init__(self, path, test_data_size, seq_length, mode="train"):
        super(CovidDataset, self).__init__()
        self.path = path
        self.test_data_size = test_data_size
        self.seq_length = seq_length
        self.mode = mode

        self.scaler = MinMaxScaler()
        self._read_file()

    def _read_file(self):
        df_all = pd.read_csv(self.path)
        df = df_all.iloc[:, 4:]
        daily_cases = df.sum(axis=0)
        daily_cases.index = pd.to_datetime(daily_cases.index)
        daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)

        self.train_data = daily_cases[:-self.test_data_size]

        self.scaler = self.scaler.fit(np.expand_dims(self.train_data, axis=1))

        if self.mode == "train":
            normal_train_data = self.scaler.transform(
                np.expand_dims(self.train_data, axis=1)).astype('float32')
            self.feature, self.label = self._create_sequences(normal_train_data)
        elif self.mode == "test":
            test_data = daily_cases[-self.test_data_size - self.seq_length + 1:]
            normal_test_data = self.scaler.transform(
                np.expand_dims(test_data, axis=1)).astype('float32')
            self.feature, self.label = self._create_sequences(normal_test_data)
        else:
            raise ValueError('Invalid Mode: Only support "train" or "test".')

    def _create_sequences(self, data):
        xs = []
        ys = []

        for i in range(len(data) - self.seq_length + 1):
            x = data[i:i + self.seq_length - 1]
            y = data[i + self.seq_length - 1]
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def postprocessing(self, data):
        result = self.scaler.inverse_transform(
            np.expand_dims(np.array(data).flatten(),
                           axis=0)).flatten().astype('int64')
        final_result = np.cumsum(
            np.concatenate([np.array(self.train_data),
                            result]))[-self.test_data_size:]
        return final_result

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return [self.feature[index], self.label[index]]

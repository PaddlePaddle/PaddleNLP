# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import time
from datetime import datetime
from multiprocessing import Pool
from tempfile import TemporaryDirectory, TemporaryFile
from paddlenlp.utils.file_lock import FileLock


class TestFileLock(unittest.TestCase):

    def test_time_lock(self):
        with TemporaryDirectory() as tempdir:
            tempfile = os.path.join(tempdir, 'download.lock')

            with FileLock(tempfile) as file_lock:

                pre_time, seconds = datetime.now(), 0

                def time_lock(*args, **kwargs):
                    file_lock.acquire()
                    time.sleep(0.5)
                    assert (datetime.now() - pre_time).seconds >= (0.5 - 1e-3)
                    pre_time = datetime.now()
                    file_lock.release()

                with Pool(4) as pool:
                    pool.starmap(time_lock, [_ for _ in range(10)])

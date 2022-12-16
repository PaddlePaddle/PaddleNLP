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


def time_lock(lock_file: str) -> datetime:
    """just sleep 1.2 seconds to test sequence timing

    Args:
        lock_file (str): the path of lock file

    Returns:
        datetime: the current datetime
    """
    with FileLock(lock_file):
        time.sleep(1.2)
    return datetime.now()


class TestFileLock(unittest.TestCase):
    def test_time_lock(self):
        """lock the time"""
        with TemporaryDirectory() as tempdir:
            lock_file = os.path.join(tempdir, "download.lock")
            pre_time, seconds = datetime.now(), 0

            with Pool(4) as pool:
                datetimes = pool.map(time_lock, [lock_file for _ in range(10)])
                datetimes.sort()

                pre_time = None
                for current_time in datetimes:
                    if pre_time is None:
                        pre_time = current_time
                    else:
                        self.assertGreater((current_time - pre_time).seconds, 1 - 1e-3)

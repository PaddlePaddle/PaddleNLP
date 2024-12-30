# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import base64
import codecs
import logging
import os
import pickle
import re
import subprocess
import time
from datetime import datetime
from enum import Enum
from logging.handlers import BaseRotatingHandler
from pathlib import Path


class DailyRotatingFileHandler(BaseRotatingHandler):
    """
    like `logging.TimedRotatingFileHandler`, but this class support multi-process
    """

    def __init__(
        self,
        filename,
        backupCount=0,
        encoding="utf-8",
        delay=False,
        utc=False,
        **kwargs
    ):
        self.backup_count = backupCount
        self.utc = utc
        self.suffix = "%Y-%m-%d"
        self.base_log_path = Path(filename)
        self.base_filename = self.base_log_path.name
        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(self.current_filename)
        BaseRotatingHandler.__init__(self, filename, "a", encoding, delay)

    def shouldRollover(self, record):
        """
        check scroll through the log
        """
        if self.current_filename != self._compute_fn():
            return True
        return False

    def doRollover(self):
        """
        scroll log
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(self.current_filename)

        if not self.delay:
            self.stream = self._open()

        self.delete_expired_files()

    def _compute_fn(self):
        """
        Calculate the log file name corresponding current time
        """
        return self.base_filename + "." + time.strftime(self.suffix, time.localtime())

    def _open(self):
        """
        open new log file
        """
        if self.encoding is None:
            stream = open(str(self.current_log_path), self.mode)
        else:
            stream = codecs.open(str(self.current_log_path), self.mode, self.encoding)

        if self.base_log_path.exists():
            try:
                if (
                    not self.base_log_path.is_symlink()
                    or os.readlink(self.base_log_path) != self.current_filename
                ):
                    os.remove(self.base_log_path)
            except OSError:
                pass

        try:
            os.symlink(self.current_filename, str(self.base_log_path))
        except OSError:
            pass
        return stream

    def delete_expired_files(self):
        """
        delete expired log files
        """
        if self.backup_count <= 0:
            return

        file_names = os.listdir(str(self.base_log_path.parent))
        result = []
        prefix = self.base_filename + "."
        plen = len(prefix)
        for file_name in file_names:
            if file_name[:plen] == prefix:
                suffix = file_name[plen:]
                if re.match(r"^\d{4}-\d{2}-\d{2}(\.\w+)?$", suffix):
                    result.append(file_name)
        if len(result) < self.backup_count:
            result = []
        else:
            result.sort()
            result = result[: len(result) - self.backup_count]

        for file_name in result:
            os.remove(str(self.base_log_path.with_name(file_name)))


def get_logger(name, file_name, without_formater=False):
    """
    get logger
    """
    log_dir = os.getenv("FD_LOG_DIR", default="log")
    is_debug = int(os.getenv("FD_DEBUG", default=0))
    logger = logging.getLogger(name)
    if is_debug:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    LOG_FILE = "{0}/{1}".format(log_dir, file_name)
    backup_count = int(os.getenv("FD_LOG_BACKUP_COUNT", 7))
    handler = DailyRotatingFileHandler(LOG_FILE, backupCount=backup_count)

    formatter = logging.Formatter(
        "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"
    )
    if not without_formater:
        handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler.propagate = False
    return logger


def str_to_datetime(date_string):
    """
    string to datetime class object
    """
    if "." in date_string:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
    else:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")


def datetime_diff(datetime_start, datetime_end):
    """
    Calculate the difference between two dates and times(s)

    Args:
        datetime_start (Union[str, datetime.datetime]): start time
        datetime_end (Union[str, datetime.datetime]): end time

    Returns:
        float: date time difference(s)
    """
    if isinstance(datetime_start, str):
        datetime_start = str_to_datetime(datetime_start)
    if isinstance(datetime_end, str):
        datetime_end = str_to_datetime(datetime_end)
    if datetime_end > datetime_start:
        cost = datetime_end - datetime_start
    else:
        cost = datetime_start - datetime_end
    return cost.total_seconds()


model_server_logger = get_logger("model_server", "infer_server.log")
http_server_logger = get_logger("http_server", "http_server.log")
data_processor_logger = get_logger("data_processor", "data_processor.log")
monitor_logger = get_logger("monitor_logger", "monitor_logger.log", True)
error_logger = get_logger("error_logger", "error_logger.log", True)

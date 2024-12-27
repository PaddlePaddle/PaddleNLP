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

import os
import threading
import time
from multiprocessing.managers import (AcquirerProxy, BaseManager, ListProxy,
                                      Value, ValueProxy)
from queue import Queue

from server.utils import get_logger

logger = get_logger("infer_server", "task_queue_manager.log")


class QueueManager(BaseManager):
    """
    base class for queue manager
    """

    pass


class TaskQueueManager(object):
    """
    task queue manager
    """

    def __init__(self, rank=0, mp_num=8, port=56666):
        """
        Initialization function, used to perform initialization
        operations when creating objects
        """
        self.max_get_num = int(os.getenv("ENGINE_MAX_NEED_NUM", 0))
        QueueManager.register('get_list')
        QueueManager.register('get_value')
        QueueManager.register('get_lock')
        QueueManager.register('get_barrier1')
        QueueManager.register('get_barrier2')
        QueueManager.register('get_queue')

        self.client_manager = QueueManager(address=('127.0.0.1', port),
                                           authkey=b'infer_queue'
                                           )
        self.client_manager.connect()
        self.list = self.client_manager.get_list()
        self.value = self.client_manager.get_value()
        self.lock = self.client_manager.get_lock()
        self.barrier1 = self.client_manager.get_barrier1()
        self.barrier2 = self.client_manager.get_barrier2()
        self.queue = self.client_manager.get_queue()
        self.mp_num = mp_num
        self.rank = rank
        self.position = 1 << rank
        self.total_num = (1 << self.mp_num) - 1
        logger.info(f"init task queue manager success, rank: {rank}")

    def empty(self):
        """
        check the queue is empty for infer

        Returns:
            bool: True if the queue is empty, otherwise False
        """
        try:
            return len(self.list) == 0
        except Exception as e:
            logger.error(f"empty function meets error: {e}")
            raise e

    def put(self, item):
        """
        put item to queue

        Args:
            item (any): the item to put into queue
        """
        self.lock.acquire()
        if 0 < self.value.get() < self.total_num:
            self.lock.release()
            while 0 < self.value.get() < self.total_num:
                time.sleep(0.001)
            logger.info("put item to queue wait finish")
            self.lock.acquire()
        if self.max_get_num <= 0 and self.value.get() == self.total_num:
            self.list[:] = []
        self.value.set(0)
        self.list.append(item)
        self.lock.release()
        logger.info("put item to queue success")

    def get(self):
        """
        get item from queue

        Returns:
            list: the item from queue
            bool: True if the queue is empty, otherwise False
        """
        input_list = []
        read_finish = False
        self.lock.acquire()
        if self.value.get() & self.position == 0 and len(self.list) > 0:
            if self.max_get_num > 0:
                input_list.extend(self.list[: self.max_get_num])
            else:
                input_list.extend(self.list[:])
            set_value = self.value.get() | self.position
            logger.info("rank: {0} set_value: {1}".format(self.rank, set_value))
            if set_value >= self.total_num:
                if self.max_get_num > 0:
                    for i in range(self.max_get_num):
                        self.list.pop(0)
                else:
                    self.list[:] = []
                set_value = 0
                read_finish = True
            self.value.set(set_value)
        self.lock.release()
        return input_list, read_finish


def launch_queue_service(port, num_workers):
    """
    Start the process communication queue service

    Args:
        port (int): the port to listen
        num_workers (int): the number of infer process
    """
    try:
        logger.info(f"start launch queue service, port:{port}")
        value = Value("i", 0)
        QueueManager.register("get_value", callable=lambda: value, proxytype=ValueProxy)
        List = list()
        QueueManager.register("get_list", callable=lambda: List, proxytype=ListProxy)
        lock = threading.Lock()
        QueueManager.register('get_lock',
                            callable=lambda: lock,
                            proxytype=AcquirerProxy)
        barrier1 = threading.Barrier(num_workers)
        QueueManager.register('get_barrier1', callable=lambda: barrier1)
        barrier2 = threading.Barrier(num_workers)
        QueueManager.register('get_barrier2', callable=lambda: barrier2)
        q = Queue()
        QueueManager.register("get_queue", callable=lambda: q)
        m = QueueManager(address=('127.0.0.1', port), authkey=b'infer_queue')
        s = m.get_server()
        logger.info("launch queue service success")
        s.serve_forever()
        logger.info("finish queue service")
    except Exception as e:
        logger.error(f"launch queue service failed, error_msg: {e}")
        raise e

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

import argparse
import os
import queue
import socket
import subprocess
import time
from collections import defaultdict
from multiprocessing import shared_memory

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from server.engine.config import Config
from server.utils import get_logger

app = FastAPI()
env_config = Config()
logger = get_logger("health_checker", "health_checker.log")


@app.get("/v2/health/ready")
def check_health():
    """
    health check interface
    """
    status, error_info = check()
    if status is True:
        logger.info("check_health: OK")
        return Response()
    else:
        logger.info("check_health: Bad")
        return JSONResponse(
                status_code=500,
                content=error_info)


@app.get("/v2/health/live")
def check_live():
    """
    health check interface
    """
    status, error_info = check()
    if status is True:
        logger.info("check_health: OK")
        return Response()
    else:
        logger.info("check_health: Bad")
        return JSONResponse(
                status_code=500,
                content=error_info)


def check_infer_engine_process():
    """
    check if infer process is alive

    return:
        status: bool, True if process is alive else False
    """
    mp_num = int(env_config.mp_num)
    for i in range(mp_num):
        try:
            infer_live_flag_shm = shared_memory.SharedMemory(name=env_config.get_unique_name("shm_flag_infer_{}_live".format(i)))
        except Exception as e:
            return False
    return True


def check():
    """
    State detection interface for inference services

    return:
        status: bool, True if process is alive else False
    """
    error_info = {}
    grpc_port = os.getenv("GRPC_PORT")

    # 1. check server is ready
    if grpc_port is not None:
        sock = socket.socket()
        try:
            sock.connect(('localhost', int(grpc_port)))
        except Exception:
            error_info["error_code"] = 1
            error_info["error_msg"] = "server is not ready"
            logger.info("server is not ready")
            return False, error_info
        finally:
            sock.close()

    # 2.check engine is ready
    is_engine_live = check_infer_engine_process()
    if is_engine_live is False:
        error_info["error_code"] = 2
        error_info["error_msg"] = "infer engine is down"
        logger.info("infer engine is down")
        return False, error_info

    engine_ready_checker = np.ndarray(engine_ready_check_flag.shape, dtype=engine_ready_check_flag.dtype,
                                      buffer=shm_engine_ready_check_flag.buf)
    if engine_ready_checker[0] == 0:
        error_info["error_code"] = 2
        error_info["error_msg"] = "infer engine is down"
        logger.info("infer engine is down")
        return False, error_info

    # check engine is hang
    engine_hang_checker = np.ndarray(engine_healthy_recorded_time.shape, dtype=engine_healthy_recorded_time.dtype,
                                buffer=shm_engine_healthy_recorded_time.buf)
    elapsed_time = time.time() - engine_hang_checker[0]
    logger.info("engine_checker elapsed time: {}".format(elapsed_time))
    if (engine_hang_checker[0]) and (elapsed_time > time_interval_threashold):
        error_info["error_code"] = 3
        error_info["error_msg"] = "infer engine hangs"
        logger.info("infer engine hangs")
        return False, error_info

    return True, error_info


def start_health_checker(http_port):
    import sys
    sys.stdout = open("log/health_http.log", 'w')
    sys.stderr = sys.stdout
    uvicorn.run(app=app, host='0.0.0.0', port=http_port, workers=1, log_level="info")


# if infer engine not update for more than 10 seconds，consider it as hang or dead
time_interval_threashold = env_config.check_health_interval
engine_healthy_recorded_time = np.zeros([1], dtype=float)

shm_engine_healthy_recorded_time = shared_memory.SharedMemory(
        create=True,
        size=engine_healthy_recorded_time.nbytes,
        name=env_config.get_unique_name("engine_healthy_recorded_time"))

engine_ready_check_flag = np.zeros([1], dtype=np.int32)
shm_engine_ready_check_flag = shared_memory.SharedMemory(
        create=True,
        size=engine_ready_check_flag.nbytes,
        name=env_config.get_unique_name("engine_ready_check_flag"))

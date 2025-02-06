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

import codecs
import json
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, deque
from datetime import datetime

import numpy as np
from server.checker import add_default_params, check_basic_params
from server.engine import engine
from server.engine.config import Config
from server.utils import error_logger, model_server_logger

import server

try:
    import triton_python_backend_utils as pb_utils
except:
    model_server_logger.warning(
        "TritonPythonModel is only available under triton inference server framework."
    )

if sys.stdout.encoding is None:
    enc = os.environ["LANG"].split(".")[1]
    sys.stdout = codecs.getwriter(enc)(sys.stdout)


class TritonConfig(Config):
    """
    Triton Inference Server config
    """
    def __init__(self, base_config):
        super().__init__()
        for k, v in base_config.__dict__.items():
            setattr(self, k, v)


class TritonTokenProcessor(engine.TokenProcessor):
    """
    initialize Triton Processor
    """
    def __init__(self, cfg, triton_server):
        super().__init__(cfg)
        self.triton_server = triton_server
        self.cached_generated_tokens = queue.Queue()
        self.token_buffer = dict()
        self.score_buffer = dict()

        self.push_mode_sender_thread = threading.Thread(target=self._push_mode_sender_thread, args=())
        self.push_mode_sender_thread.daemon = True
        self.push_mode_sender_thread.start()

    def _push_mode_sender_thread(self):
        """
        push mode sender thread
        """
        while True:
            try:
                batch_result = self.cached_generated_tokens.get()
                for result in batch_result:
                    req_id = result["req_id"]
                    is_end = result.get("is_end", 0)
                    return_all_tokens = result.get("return_all_tokens", False)
                    if is_end == 0 and (return_all_tokens or self.cfg.disable_streaming):
                        continue
                    if return_all_tokens and "topk_tokens" in result:
                        del result["topk_tokens"]
                    result = self.triton_server.data_processor.process_response(result)
                    if "usage" in result:
                        result["usage"]["prompt_tokens"] = self.triton_server.task_info[req_id]["prompt_tokens"]
                    model_server_logger.debug(f"Send result to client under push mode: {result}")
                    with self.triton_server.thread_lock:
                        _send_result([result], self.triton_server.response_sender[req_id], is_end)
                        if is_end == 1:
                            del self.triton_server.response_sender[req_id]
                            del self.triton_server.task_info[req_id]
                            self.triton_server._update_metrics()
            except Exception as e:
                    model_server_logger.error("Unexcepted error happend: {}, {}".format(e, str(traceback.format_exc())))

    def _cache_special_tokens(self, batch_result):
        for i in range(len(batch_result)):
            is_end = batch_result[i].get("is_end", 0)
            token_ids = batch_result[i]["token_ids"]
            if is_end != 1:
                if batch_result[i]["req_id"] not in self.token_buffer:
                    self.token_buffer[batch_result[i]["req_id"]] = list()
                    self.score_buffer[batch_result[i]["req_id"]] = list()
                self.token_buffer[batch_result[i]["req_id"]].extend(token_ids)
                self.score_buffer[batch_result[i]["req_id"]].extend(batch_result[i].get("token_scores", []))
                batch_result[i]["token_ids"] = []
                if "token_scores" in batch_result[i]:
                    batch_result[i]["token_scores"] = []
            else:
                if batch_result[i]["req_id"] in self.token_buffer:
                    batch_result[i]["token_ids"] = self.token_buffer[batch_result[i]
                        ["req_id"]] + batch_result[i]["token_ids"]
                    del self.token_buffer[batch_result[i]["req_id"]]
                    if "token_scores" in batch_result[i]:
                        batch_result[i]["token_scores"] = self.score_buffer[batch_result[i]
                            ["req_id"]] + batch_result[i]["token_scores"]
                    del self.score_buffer[batch_result[i]["req_id"]]

    def postprocess(self, batch_result, exist_finished_task=False):
        """
        single postprocess for triton
        """
        try:
            self._cache_special_tokens(batch_result)
            self.cached_generated_tokens.put(batch_result)
        except Exception as e:
            model_server_logger.info(
                "Unexcepted problem happend while process output token: {}, {}"
                .format(e, str(traceback.format_exc())))


class TritonServer(object):
    """
    Triton Server
    """

    def initialize(self, args):
        """
        Triton initialization
        """
        # start health checker
        use_custom_health_checker = int(os.getenv("USE_CUSTOM_HEALTH_CHECKER", 1))
        # if set USE_CUSTOM_HEALTH_CHECKER=1, use custom health checker, need set --allow-http=false
        # else use tritonserver's health checker, need set --http-port=${HTTP_PORT}
        if use_custom_health_checker:
            http_port = os.getenv("HTTP_PORT")
            if http_port is None:
                raise Exception("HTTP_PORT must be set")
            from server.triton_server_helper import start_health_checker
            multiprocessing.Process(target=start_health_checker, args=(int(http_port), )).start()
            time.sleep(1)

        model_config = json.loads(args["model_config"])
        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config)
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(args["model_name"]))

        # add metrics，use METRICS_PORT get server metrics
        self.metric_family = pb_utils.MetricFamily(
            name="inference_server_metrics",
            description="Metrics for monitoring inference server status",
            kind=pb_utils.MetricFamily.
            GAUGE,
        )
        self.metrics = {
            "batch_size":
            self.metric_family.Metric(labels={"batch_size": "batch_size"}),
            "block_num":
            self.metric_family.Metric(labels={"block_num": "block_num"}),
            "max_batch_size":
            self.metric_family.Metric(
                labels={"max_batch_size": "max_batch_size"}),
            "max_block_num":
            self.metric_family.Metric(
                labels={"max_block_num": "max_block_num"}),
            "available_resource":
            self.metric_family.Metric(
                labels={"available_resource": "available_resource"}),
        }

        # response_sender thread lock
        self.thread_lock = threading.Lock()

        base_config = Config()
        self.cfg = TritonConfig(base_config)
        self.cfg.print(file="log/fastdeploy_init.info")

        # init engine
        self.token_processor = TritonTokenProcessor(self.cfg, self)
        self.engine = engine.Engine(self.cfg, self.token_processor)
        model_server_logger.info("Creat engine...")
        self.engine.start()
        model_server_logger.info("Create engine success")

        self._initialize_push_mode()
        model_server_logger.info("Init triton server success")


    def execute(self, requests):
        """
        Triton service main function,
        handling requests received by the Triton framework
        """
        if len(requests) != 1:
            raise pb_utils.TritonModelException(
                "Only support batch=1, but now it's {}.".format(len(requests)))
        request = requests[0]
        current_response_sender = request.get_response_sender()
        request_tensor = pb_utils.get_input_tensor_by_name(request, "IN")
        tasks = json.loads(request_tensor.as_numpy()[0])

        model_server_logger.info(f"receive task: {tasks}")
        self._process_task_push_mode(tasks, current_response_sender)
        self._update_metrics()

    def finalize(self):
        """
        Triton service exit function
        """
        model_server_logger.info("Triton service will be terminated...")
        wait_time = 300
        while not self.engine.all_tasks_finished():
            if wait_time <= 0:
                model_server_logger.warning(f"Ignore the unfinished tasks, force to stop.")
                break
            model_server_logger.info(f"There's unfinished tasks, wait {wait_time}...")
            wait_time -= 5
            time.sleep(5)
        model_server_logger.info("Terminate the engine now.")
        self.enable_insert_task_push_mode = False
        time.sleep(1)
        del self.engine
        if hasattr(self, "http_process"):
            self.http_process.kill()
        model_server_logger.info("Triton service is terminated!")

    def _initialize_push_mode(self):
        from server.data.processor import DataProcessor
        self.data_processor = DataProcessor()
        model_server_logger.info("create data processor success")

        if self.cfg.push_mode_http_port < 0:
            model_server_logger.info("HTTP server for push mode is disabled.")
        else:
            model_server_logger.info("launch http server...")

            current_dir_path = os.path.split(os.path.abspath(__file__))[0]
            http_py_file = "app.py"
            http_py_path = os.path.join(current_dir_path, "http_server", http_py_file)
            http_cmd = f"python3 {http_py_path} --port={self.cfg.push_mode_http_port} " \
                     f"--workers={self.cfg.push_mode_http_workers} >log/launch_http.log 2>&1"

            model_server_logger.info(f"Launch HTTP server for push mode, command:{http_cmd}")
            self.http_process = subprocess.Popen(http_cmd, shell=True, preexec_fn=os.setsid)
            time.sleep(3)
            exit_code = self.http_process.poll()
            if exit_code is None:
                http_url = f"http://127.0.0.1:{self.cfg.push_mode_http_port}/v1/chat/completions"
                model_server_logger.info(f"Launch HTTP server for push mode success, http_url:{http_url}")
            else:
                error_msg = "\n Launch HTTP service for push mode failed in 3 seconds. " \
                            "Please check log/launch_http.log file \n"
                model_server_logger.error(error_msg)
            model_server_logger.info("init push server success")

            self.response_sender = dict()
            self.task_info = dict()
            self.cached_task_deque = deque()
            self.enable_insert_task_push_mode = True
            self.insert_task_to_engine_thread = threading.Thread(
                target=self._insert_task_push_mode, args=())
            self.insert_task_to_engine_thread.daemon = True
            self.insert_task_to_engine_thread.start()

    def _process_task_push_mode(self, tasks, current_response_sender):
        """
        check request and insert into cached_task_deque

        Args:
            tasks (list): list of request
            current_response_sender: response sender for current request
        """
        try:
            tik = time.time()
            req_id = tasks[0]["req_id"]
            cached_task_num = len(self.cached_task_deque)
            if cached_task_num >= self.cfg.max_cached_task_num:
                error_msg = f"cached task num ({cached_task_num}) exceeds " \
                            f"the limit ({self.cfg.max_cached_task_num})"
                _send_error(error_msg, current_response_sender, req_id=req_id)
                return

            if not tasks or len(tasks) != 1 or not tasks[0]:
                error_msg = f"request data should not be empty and query " \
                            f"num {len(tasks)} should be 1"
                _send_error(error_msg, current_response_sender, req_id=req_id)
                return

            task = tasks[0]
            task["preprocess_start_time"] = datetime.now()

            error_msg = check_basic_params(task)
            if error_msg != []:
                _send_error(error_msg, current_response_sender, req_id=req_id)
                return

            task_id = task["req_id"]
            with self.thread_lock:
                if task_id in self.response_sender:
                    error_msg =  f"The req_id {task_id} already exists in the current batch, " \
                                    f"the current request will be ignored."
                    _send_error(error_msg, current_response_sender, req_id=req_id)
                    return

            task = add_default_params(task)

            if int(task.get("enable_text_truncate", 1)):
                real_seq_len = self.cfg.max_seq_len - task.get("max_dec_len", 800)
                task = self.data_processor.process_request(task, max_seq_len=real_seq_len)
            else:
                task = self.data_processor.process_request(task)

            input_ids_len = len(task["input_ids"])
            if "max_dec_len" not in task:
                task["max_dec_len"] = min(self.cfg.max_seq_len - input_ids_len, self.cfg.dec_len_limit)
            min_dec_len = task["min_dec_len"]
            if input_ids_len + min_dec_len >= self.cfg.max_seq_len:
                error_msg = f"Input text is too long, input_ids_len ({input_ids_len}) " \
                            f"+ min_dec_len ({min_dec_len}) >= max_seq_len "
                _send_error(error_msg, current_response_sender, req_id=req_id)
                return

            if input_ids_len > self.cfg.seq_len_limit:
                error_msg = f"Length of input token({input_ids_len}) exceeds the limit MAX_SEQ_LEN({self.cfg.seq_len_limit})."
                _send_error(error_msg, current_response_sender, req_id=req_id)
                return
            if task["max_dec_len"] > self.cfg.dec_len_limit:
                error_msg = f"The parameter max_dec_len({task['max_dec_len']}) exceeds the limit MAX_DEC_LEN({self.cfg.dec_len_limit})."
                _send_error(error_msg, current_response_sender, req_id=req_id)
                return

            required_block_num = self.engine.resource_manager.get_required_block_number(input_ids_len)
            if required_block_num > self.engine.resource_manager.total_block_number():
                error_msg = f"The input task required resources is exceed the limit, task={task}."
                _send_error(error_msg, current_response_sender, req_id=req_id)
                return

            with self.thread_lock:
                self.response_sender[task_id] = current_response_sender
                self.task_info[task_id] = {"prompt_tokens": input_ids_len}

            task["preprocess_end_time"] = datetime.now()
            self.cached_task_deque.appendleft(task)
            tok = time.time()
            model_server_logger.info(f"cache task with req_id ({task_id}), "
                                     f"cost time: {tok-tik}s, cached_task_num: {len(self.cached_task_deque)}.")
            model_server_logger.debug(f"cache task: {task}")
        except Exception as e:
            error_msg = "Unexcepted promblem happend while insert new task to server task queue: {}, {}".format(
                e, str(traceback.format_exc()))
            _send_error(error_msg, current_response_sender)

    def _insert_task_push_mode(self):
        """
        Insert task to engine thread, monitor cached_task_deque.
        if the engine has resource, insert task to engine
        """
        try:
            while self.enable_insert_task_push_mode:
                if not hasattr(self, "engine") or self.engine is None:
                    time.sleep(0.1)
                    continue
                if self.engine.available_batch() == 0:
                    time.sleep(0.001)
                    continue
                if len(self.cached_task_deque) == 0:
                    time.sleep(0.001)
                    continue
                if not self.engine.is_queue_empty():
                    time.sleep(0.001)
                    continue

                i_bs = 0
                for _ in range(self.cfg.max_prefill_batch):
                    if len(self.cached_task_deque) == 0:
                        break
                    if self.engine.available_batch() == 0:
                        break
                    while i_bs < self.cfg.max_batch_size:
                        if self.engine.task_is_finished(i_bs):
                            break
                        i_bs += 1
                    if i_bs >= self.cfg.max_batch_size:
                        break
                    input_token_num = len(self.cached_task_deque[-1]["input_ids"])
                    if not self.engine.is_resource_sufficient(input_token_num):
                        break
                    task = self.cached_task_deque.pop()
                    try:
                        self.engine.insert_tasks([task])
                    except Exception as e:
                        err_msg = "Error happend while insert task to engine: {}, {}.".format(
                            e, str(traceback.format_exc()))
                        with self.thread_lock:
                            _send_result({"error_msg": err_msg},
                                        self.response_sender[task["req_id"]], 1)
                            del self.response_sender[task["req_id"]]
            model_server_logger.info("finish insert_task_push_mode thread")
        except Exception as e:
            model_server_logger.error("insert_task_push_mode thread exit "
                                      f"unexpectedly, {e}. {str(traceback.format_exc())}")

    def _update_metrics(self):
        """
        update metrics
        """
        block_num = self.engine.available_block_num()
        batch_size = self.engine.available_batch()
        self.metrics["block_num"].set(block_num)
        self.metrics["max_batch_size"].set(self.cfg.max_batch_size)
        self.metrics["batch_size"].set(self.cfg.max_batch_size - batch_size)
        self.metrics["max_block_num"].set(self.cfg.max_block_num)
        self.metrics["available_resource"].set(block_num * 1.0 /
                                               self.cfg.max_block_num)

    def _get_current_server_info(self):
        """
        get server info
        """
        available_batch_size = min(self.cfg.max_prefill_batch,
                                   self.engine.available_batch())
        available_block_num = self.engine.available_block_num()
        server_info = {
            "block_size": int(self.cfg.block_size),
            "block_num": int(available_block_num),
            "dec_token_num": int(self.cfg.dec_token_num),
            "available_resource":
            1.0 * available_block_num / self.cfg.max_block_num,
            "max_batch_size": int(available_batch_size),
        }
        return server_info


def _send_result(result_dict, sender, end_flag=0):
    """
    Send inference result

    Args:
        result_dict (dict): result of inference
        sender (grpc.aio.ServerReaderWriter): gRPC ServerReaderWriter object.
        end_flag (int, optional): flag of end. Defaults to 0.
    """
    response = None
    if result_dict:
        result_dict = json.dumps(result_dict)
        end_output = pb_utils.Tensor("OUT",
                                     np.array([result_dict], dtype=np.object_))
        response = pb_utils.InferenceResponse(output_tensors=[end_output])
    if response is None and end_flag == 0:
        return
    sender.send(response, flags=end_flag)

def _send_error(error_msg, sender, error_code=200, req_id=None):
    """
    Send error inference result

    Args:
        error_msg (str): error message
        sender (grpc.aio.ServerReaderWriter): gRPC ServerReaderWriter object.
        error_code (int, optional): error code. Defaults to 200.
        req_id (str, optional): request id. Defaults to None
    """
    if not isinstance(error_msg, str):
        error_msg = str(error_msg)
    error_info = {"req_id": req_id, "error_msg": error_msg, "error_code": error_code, "version": "4.6", "timestamp": time.time()}
    error_logger.info(f"{error_info}")
    model_server_logger.error(error_msg)
    _send_result(error_info, sender, 1)


TritonPythonModel = TritonServer

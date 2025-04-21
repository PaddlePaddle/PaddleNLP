""" 
warmup client 
"""
import argparse
import json
import os
import queue
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

parse = argparse.ArgumentParser(description='A program to generate config')
parse.add_argument('--batch_size',
                   '-bs',
                   type=int,
                   help='batch size',
                   default=2)
parse.add_argument('--input_length',
                   type=int,
                   help='input length',
                   default=6144)
args = parse.parse_args()
batch_size = args.batch_size
input_length = args.input_length

GRPC_PORT = int(os.getenv('GRPC_PORT', default=8811))


class UserData:
    """
    Init UserData.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    """
    Asynchronous callback function to handle request results.

    Args:
        user_data (Any): User-defined data to be passed to the callback.
        result (Any): The result of the request.
        error (Any): An error that occurred during the request, if any.

    Returns:
        None
    """
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


model_name = "model"
inputs = [grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.object_))]
outputs = [grpcclient.InferRequestedOutput("OUT")]

user_data = UserData()


def has_list(obj):
    """
    has_list func
    
    Args:
        obj: object.
    
    Returns:
        True or False.
    
    """
    if isinstance(obj, list):
        return True
    if isinstance(obj, dict):
        for value in obj.values():
            if has_list(value):
                return True
    return False


with grpcclient.InferenceServerClient(url="localhost:{}".format(GRPC_PORT),
                                      verbose=False) as triton_client:
    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))

    assert input_length > 2, "input tokens num should be larger than 2"
    input_ids = [855] * input_length
    input_ids[0] = 128803 
    input_ids[-1] = 201 
    input_ids[-2] = 128798
    input_ids[-3] = 128804

    data1 = {
        "text": "<cls>" + "北" + "<sep>",
        "input_ids": input_ids,
        "req_id": 0,
        "seq_len": 1,
        "min_dec_len": 1,
        "max_dec_len": 256,
        "penalty_score": 1.0,
        "temperature": 0.8,
        "topp": 0,
        "frequency_score": 0.1,
        # "eos_token_ids": [
        #     2,
        # ],
        "presence_score": 0.0
    }
    data_list = [data1.copy() for _ in range(batch_size)]
    for idx, d in enumerate(data_list):
        d['req_id'] = idx

    # Send specified many requests in parallel
    for i in range(1):
        req_dict = json.dumps(data_list)
        in_data = np.array([req_dict], dtype=np.object_)
        inputs[0].set_data_from_numpy(in_data)

        triton_client.async_stream_infer(model_name=model_name,
                                         inputs=inputs,
                                         request_id="{}".format(i),
                                         outputs=outputs)
        # Retrieve results...
        while True:
            data_item = user_data._completed_requests.get(timeout=1200)
            if type(data_item) == InferenceServerException:
                print('Exception:', 'status', data_item.status(), 'msg',
                      data_item.message())

            else:
                request_id = data_item.get_response().id
                results = data_item.as_numpy("OUT")
                result = results[0]
                json_res = json.loads(result)
                if has_list(json_res):
                    data = json_res[0]
                    print(data)
                    if data["is_end"] == 1:
                        break
        os._exit(0)

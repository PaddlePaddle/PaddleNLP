# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.inference as paddle_infer

from paddlenlp.utils.downloader import get_path_from_url_with_filelock


def checkmodel(model_dir, model_name):
    if not os.path.exists(os.path.join(model_dir, model_name, model_name + ".pdmodel")):
        model_url = "https://bj.bcebos.com/v1/paddledet/models/dpt_hybrid.zip"
        get_path_from_url_with_filelock(model_url, root_dir=model_dir)


class MidasInference:
    def __init__(self, model_dir, model_name="dpt_hybrid", batchsize=8, device="GPU", run_mode="paddle"):
        checkmodel(model_dir, model_name)
        model_file = os.path.join(model_dir, model_name, model_name + ".pdmodel")
        params_file = os.path.join(model_dir, model_name, model_name + ".pdiparams")
        config = paddle_infer.Config(model_file, params_file)
        self.batchsize = batchsize
        if device == "GPU":
            # initial GPU memory(M), device ID
            config.enable_use_gpu(200, 0)
            # optimize graph and fuse op
            config.switch_ir_optim(True)
        elif device == "XPU":
            if config.lite_engine_enabled():
                config.enable_lite_engine()
            config.enable_xpu(10 * 1024 * 1024)
        elif device == "NPU":
            if config.lite_engine_enabled():
                config.enable_lite_engine()
            config.enable_npu()
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(4)

        precision_map = {
            "trt_int8": paddle_infer.Config.Precision.Int8,
            "trt_fp32": paddle_infer.Config.Precision.Float32,
            "trt_fp16": paddle_infer.Config.Precision.Half,
        }
        if run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=(1 << 25) * batchsize,
                max_batch_size=batchsize,
                min_subgraph_size=3,
                precision_mode=precision_map[run_mode],
                use_static=False,
                use_calib_mode=False,
            )

        # disable print log when predict
        config.disable_glog_info()
        # enable shared memory
        config.enable_memory_optim()
        # disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle_infer.create_predictor(config)

    def predict(self, inputs):

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])

        input_handle.copy_from_cpu(inputs)
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()  # numpy.ndarray
        return output_data

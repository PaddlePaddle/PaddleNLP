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
from typing import Dict, List, Tuple

import numpy as np
import paddle
import six
from psutil import cpu_count

from paddlenlp.transformers import AutoTokenizer


class InferBackend(object):
    def __init__(
        self,
        model_path,
        batch_size=32,
        device="cpu",
        cpu_backend="mkldnn",
        precision_mode="fp32",
        set_dynamic_shape=False,
        shape_info_file="shape_info.txt",
        num_threads=10,
    ):
        """
        Args:
            model_path (str): The model path for deployment.
            batch_size (int): Batch size of input, the default is 32.
            device (str): The deployed device can be set to cpu, gpu or xpu, the default is cpu.
            cpu_backend (str): Inference backend when deploy on cpu, which can be mkldnn or onnxruntime,
                                the default is mkldnn.
            precision_mode (str): Inference precision, which can be fp32, fp16 or int8, the default is fp32.
            set_dynamic_shape (bool): Whether to set_dynamic_shape for Inference-TRT, the default is False.
            shape_info_file (str): When set_dynamic_shape is enabled, the file name of shape_info is stored,
                                    the default is shape_info.txt.
            num_threads (int): Number of cpu threads during inference, the default is 10.
        """
        precision_mode = precision_mode.lower()
        use_fp16 = precision_mode == "fp16"
        use_quantize = precision_mode == "int8"
        model_path = self.model_path_correction(model_path)
        # Check if the model is a quantized model
        is_int8_model = self.paddle_quantize_model(model_path)
        print(">>> [InferBackend] Creating Engine ...")

        self.predictor_type = "ONNXRuntime"
        if is_int8_model or device == "xpu" or device == "cpu" and not use_quantize:
            self.predictor_type = "Inference"

        if self.predictor_type == "Inference":
            from paddle import inference

            config = paddle.inference.Config(model_path + ".pdmodel", model_path + ".pdiparams")
            # quantized model on GPU
            if device == "gpu":
                config.enable_use_gpu(100, 0)

                precision_type = inference.PrecisionType.Float32
                if is_int8_model:
                    print(">>> [InferBackend] INT8 inference on GPU ...")
                    precision_type = inference.PrecisionType.Int8

                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=precision_type,
                    max_batch_size=batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False,
                )
                if set_dynamic_shape:
                    config.collect_shape_range_info(shape_info_file)
                else:
                    config.enable_tuned_tensorrt_dynamic_shape(shape_info_file, True)
            elif device == "cpu":
                config.disable_gpu()
                config.switch_ir_optim(True)
                if cpu_backend == "mkldnn":
                    config.enable_mkldnn()
                    if use_fp16:
                        print(">>> [InferBackend] FP16 inference on CPU ...")
                        config.enable_mkldnn_bfloat16()
                    if is_int8_model:
                        print(">>> [InferBackend] INT8 inference on CPU ...")
                        config.enable_mkldnn_int8()
                elif cpu_backend == "onnxruntime":
                    if use_fp16:
                        print(">>> [InferBackend] FP16 is not supported in ORT backend ...")
                    config.enable_onnxruntime()
                    config.enable_ort_optimization()
            elif device == "xpu":
                print(">>> [InferBackend] Inference on XPU ...")
                config.enable_xpu(100)

            config.enable_memory_optim()
            config.set_cpu_math_library_num_threads(num_threads)
            config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
            self.predictor = paddle.inference.create_predictor(config)
            self.input_names = [name for name in self.predictor.get_input_names()]
            self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
            self.output_handles = [
                self.predictor.get_output_handle(name) for name in self.predictor.get_output_names()
            ]
        else:

            import onnxruntime as ort
            import paddle2onnx

            onnx_model = paddle2onnx.command.c_paddle_to_onnx(
                model_file=model_path + ".pdmodel",
                params_file=model_path + ".pdiparams",
                opset_version=13,
                enable_onnx_checker=True,
            )

            deploy_onnx_model = onnx_model
            providers = ["CUDAExecutionProvider"]

            # Can not use ONNXRuntime dynamic quantize when deploy on GPU
            if device == "gpu" and use_quantize:
                print(
                    ">>> [InferBackend] It is a FP32 model, and dynamic quantization "
                    "is not supported on gpu, use FP32 to inference here ..."
                )
                use_quantize = False

            if use_fp16 and use_quantize:
                print(">>> [InferBackend] Both FP16 and Int8 are enabled, use FP16 to inference here ...")
                use_quantize = False

            if use_fp16:
                import onnx
                from onnxconverter_common import float16

                deploy_onnx_model = "fp16_model.onnx"
                onnx_model_proto = onnx.ModelProto()
                onnx_model_proto.ParseFromString(onnx_model)
                trans_model = float16.convert_float_to_float16(onnx_model_proto, keep_io_types=True)
                onnx.save_model(trans_model, deploy_onnx_model)
                print(">>> [InferBackend] FP16 inference on GPU ...")

            if use_quantize:
                from onnxruntime.quantization import quantize_dynamic

                deploy_onnx_model = "dynamic_quantize_model.onnx"
                float_onnx_file = "model.onnx"
                with open(float_onnx_file, "wb") as f:
                    f.write(onnx_model)
                quantize_dynamic(float_onnx_file, deploy_onnx_model)
                providers = ["CPUExecutionProvider"]
                print(">>> [InferBackend] INT8 inference on CPU ...")

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = num_threads
            self.predictor = ort.InferenceSession(deploy_onnx_model, sess_options=sess_options, providers=providers)
            input_name = self.predictor.get_inputs()[0].name
            self.input_handles = [input_name]
            self.output_handles = []
        print(">>> [InferBackend] Engine Created ...")

    def model_path_correction(self, model_path):
        if os.path.isfile(model_path + ".pdmodel"):
            return model_path
        new_model_path = None
        for file in os.listdir(model_path):
            if file.count(".pdmodel"):
                filename = file[:-8]
                new_model_path = os.path.join(model_path, filename)
                return new_model_path
        assert new_model_path is not None, "Can not find model file in your path."

    def paddle_quantize_model(self, model_path):
        model = paddle.jit.load(model_path)
        program = model.program()
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type.count("quantize"):
                    return True
        return False

    def infer(self, input_dict: dict):
        if self.predictor_type == "Inference":
            for idx, input_name in enumerate(self.input_names):
                self.input_handles[idx].copy_from_cpu(input_dict[input_name])
            self.predictor.run()
            output = [output_handle.copy_to_cpu() for output_handle in self.output_handles]
            return output

        result = self.predictor.run(None, input_dict)
        return result


def seq_cls_print_ret(infer_result: List[np.ndarray], input_data: Tuple[List[str], List[str]]):
    label_list = ["entailment", "neutral", "contradiction"]
    label = infer_result["label"].squeeze().tolist()
    confidence = infer_result["confidence"].squeeze().tolist()
    for i in range(len(label)):
        print("input data:", input_data[0][i], input_data[1][i])
        print("seq cls result:")
        print("label:", label_list[label[i]], "  confidence:", confidence[i])
        print("-----------------------------")


class ErnieMPredictor(object):
    def __init__(self, args):
        if not isinstance(args.device, six.string_types):
            print(">>> [InferBackend] The type of device must be string, but the type you set is: ", type(args.device))
            exit(0)
        args.device = args.device.lower()
        if args.device not in ["cpu", "gpu", "xpu"]:
            print(">>> [InferBackend] The device must be cpu or gpu, but your device is set to:", type(args.device))
            exit(0)

        self.task_name = args.task_name
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        if args.task_name == "seq_cls":
            self.label_names = []
            self.preprocess = self.seq_cls_preprocess
            self.postprocess = self.seq_cls_postprocess
            self.printer = seq_cls_print_ret
        else:
            print("[ErniePredictor]: task_name only support seq_cls now.")
            exit(0)

        self.max_seq_length = args.max_seq_length

        if args.device == "cpu":
            args.set_dynamic_shape = False
            args.shape_info_file = None
            args.batch_size = 32
        if args.device == "gpu":
            args.num_threads = cpu_count(logical=False)
        self.inference_backend = InferBackend(
            args.model_path,
            batch_size=args.batch_size,
            device=args.device,
            precision_mode=args.precision_mode,
            set_dynamic_shape=args.set_dynamic_shape,
            shape_info_file=args.shape_info_file,
            num_threads=args.num_threads,
        )
        if args.set_dynamic_shape:
            # If set_dynamic_shape is turned on, all required dynamic shapes will be
            # automatically set according to the batch_size and max_seq_length.
            self.set_dynamic_shape(args.max_seq_length, args.batch_size)
            exit(0)

    def seq_cls_preprocess(self, input_data: Tuple[List[str], List[str]]) -> Dict[str, np.ndarray]:
        # tokenizer + pad
        data = self.tokenizer(
            *input_data, max_length=self.max_seq_length, padding=True, truncation=True, return_token_type_ids=False
        )
        input_ids = data["input_ids"]
        return {
            "input_ids": np.array(input_ids, dtype="int64"),
        }

    def seq_cls_postprocess(self, infer_data: List[np.ndarray], input_data: Tuple[List[str], List[str]]):
        logits = np.array(infer_data[0])
        max_value = np.max(logits, axis=1, keepdims=True)
        exp_data = np.exp(logits - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {"label": probs.argmax(axis=-1), "confidence": probs.max(axis=-1)}
        return out_dict

    def set_dynamic_shape(self, max_seq_length, batch_size):
        # The dynamic shape info required by TRT is automatically generated
        # according to max_seq_length and batch_size and stored in shape_info.txt
        min_batch_size, max_batch_size, opt_batch_size = 1, batch_size, batch_size
        min_seq_len, max_seq_len, opt_seq_len = 2, max_seq_length, max_seq_length
        batches = [
            {
                "input_ids": np.zeros([min_batch_size, min_seq_len], dtype="int64"),
                "token_type_ids": np.zeros([min_batch_size, min_seq_len], dtype="int64"),
            },
            {
                "input_ids": np.zeros([max_batch_size, max_seq_len], dtype="int64"),
                "token_type_ids": np.zeros([max_batch_size, max_seq_len], dtype="int64"),
            },
            {
                "input_ids": np.zeros([opt_batch_size, opt_seq_len], dtype="int64"),
                "token_type_ids": np.zeros([opt_batch_size, opt_seq_len], dtype="int64"),
            },
        ]
        for batch in batches:
            self.inference_backend.infer(batch)
        print("[InferBackend] Set dynamic shape finished, please close set_dynamic_shape and restart.")

    def infer(self, data: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.inference_backend.infer(data)

    def predict(self, input_data: Tuple[List[str], List[str]]):
        preprocess_result = self.preprocess(input_data)
        infer_result = self.infer(preprocess_result)
        result = self.postprocess(infer_result, input_data)
        self.printer(result, input_data)
        return result

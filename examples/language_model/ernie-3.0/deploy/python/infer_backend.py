# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from pathlib import Path
import six


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(
        filename.suffix)


def optimize(onnx_model_path: Path) -> Path:
    from onnxruntime import InferenceSession, SessionOptions
    opt_model_path = generate_identified_filename(onnx_model_path, "-optimized")
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    _ = InferenceSession(onnx_model_path.as_posix(), sess_option)
    return opt_model_path


def dynamic_quantize(input_float_model: str, dynamic_quantized_model: str):
    import onnx
    from onnxruntime.quantization import QuantizationMode, quantize_dynamic
    onnx_model = Path(input_float_model).absolute()
    optimized_output = optimize(onnx_model)
    quantize_dynamic(optimized_output, dynamic_quantized_model)


class InferBackend(object):
    def __init__(
            self,
            model_path,
            batch_size=32,
            device='cpu',
            use_int8=False,
            collect_shape=False,
            num_threads=10,
            use_inference=False,  # Users can specify to use paddleinference for inference
            use_trt=False):
        self.device = device
        self.use_int8 = use_int8
        self.use_inference = use_inference
        if not isinstance(device, six.string_types):
            print(
                ">>> [InferBackend] The type of device must be string, but the type you set is: ",
                type(device))
            exit(0)
        device = device.lower()
        if device not in ['cpu', 'gpu']:
            print(
                ">>> [InferBackend] The device must be cpu or gpu, but your device is set to:",
                type(device))
            exit(0)

        print(">>> [InferBackend] creat engine ...")
        if use_inference:
            print("[InferBackend] use PaddleInference to infer ...")
            from paddle import inference
            import paddle
            config = paddle.inference.Config(model_path + ".pdmodel",
                                             model_path + ".pdiparams")
            if device == "gpu":
                # set GPU configs accordingly
                config.enable_use_gpu(100, 0)
                paddle.set_device("gpu")
            elif device == "cpu":
                config.disable_gpu()
                config.switch_ir_optim(True)
                config.enable_mkldnn()
                config.set_cpu_math_library_num_threads(num_threads)
                paddle.set_device("cpu")

            if use_trt:
                assert device == "gpu", "when use_trt, the device must be gpu."
                if use_int8:
                    config.enable_tensorrt_engine(
                        workspace_size=1 << 30,
                        precision_mode=inference.PrecisionType.Int8,
                        max_batch_size=batch_size,
                        min_subgraph_size=5,
                        use_static=False,
                        use_calib_mode=False)
                else:
                    config.enable_tensorrt_engine(
                        workspace_size=1 << 30,
                        precision_mode=inference.PrecisionType.Float32,
                        max_batch_size=batch_size,
                        min_subgraph_size=5,
                        use_static=False,
                        use_calib_mode=False)
                shape_file = "shape_info.txt"
                if collect_shape:
                    config.collect_shape_range_info(shape_file)
                else:
                    config.enable_tuned_tensorrt_dynamic_shape(shape_file, True)

            config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
            self.predictor = paddle.inference.create_predictor(config)
            self.input_handles = [
                self.predictor.get_input_handle(name)
                for name in self.predictor.get_input_names()
            ]
            self.output_handles = [
                self.predictor.get_output_handle(name)
                for name in self.predictor.get_output_names()
            ]
            print(">>> [InferBackend] PaddleInference engine created ...")
            return

        if device == 'gpu' and use_int8:
            from paddle import inference
            import paddle
            config = paddle.inference.Config(model_path + ".pdmodel",
                                             model_path + ".pdiparams")
            config.enable_use_gpu(100, 0)
            paddle.set_device("gpu")
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                precision_mode=inference.PrecisionType.Int8,
                max_batch_size=batch_size,
                min_subgraph_size=5,
                use_static=False,
                use_calib_mode=False)

            shape_file = "shape_info.txt"
            if collect_shape:
                config.collect_shape_range_info(shape_file)
            else:
                config.enable_tuned_tensorrt_dynamic_shape(shape_file, True)
            config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
            self.predictor = paddle.inference.create_predictor(config)
            self.input_handles = [
                self.predictor.get_input_handle(name)
                for name in self.predictor.get_input_names()
            ]
            self.output_handles = [
                self.predictor.get_output_handle(name)
                for name in self.predictor.get_output_names()
            ]
        else:
            import paddle2onnx
            import onnxruntime as ort
            import copy
            import os
            float_onnx_file = "model.onnx"
            file_name = model_path.split('/')[-1]
            model_path = model_path[:-1 * len(file_name)]
            paddle2onnx.py_program2onnx(
                model_dir=model_path,
                model_filename=file_name + ".pdmodel",
                params_filename=file_name + ".pdiparams",
                save_file=float_onnx_file,
                opset_version=13,
                enable_onnx_checker=True)
            dynamic_quantize_onnx_file = copy.copy(float_onnx_file)
            providers = ['CUDAExecutionProvider']
            if device == 'cpu' and use_int8:
                dynamic_quantize_onnx_file = "dynamic_quantize_model.onnx"
                dynamic_quantize(float_onnx_file, dynamic_quantize_onnx_file)
                providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.optimized_model_filepath = "./optimize_model.onnx"
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
            self.predictor = ort.InferenceSession(
                dynamic_quantize_onnx_file,
                sess_options=sess_options,
                providers=providers)
            input_name1 = self.predictor.get_inputs()[0].name
            input_name2 = self.predictor.get_inputs()[1].name
            self.input_handles = [input_name1, input_name2]
            self.output_handles = []
        print(">>> [InferBackend] engine created ...")

    def infer(self, data):
        if self.use_inference or self.device == 'gpu' and self.use_int8:
            for input_field, input_handle in zip(data, self.input_handles):
                input_handle.copy_from_cpu(input_field)
            self.predictor.run()
            output = [
                output_handle.copy_to_cpu()
                for output_handle in self.output_handles
            ]
            return output
        input_dict = {}
        for input_field, input_handle in zip(data, self.input_handles):
            input_dict[input_handle] = input_field
        result = self.predictor.run(None, input_dict)
        return result

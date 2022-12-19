# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import abc
import math
import os
from abc import abstractmethod
from multiprocessing import cpu_count

import paddle
from paddle.dataset.common import md5file

from ..utils.env import PPNLP_HOME
from ..utils.log import logger
from .utils import cut_chinese_sent, download_check, download_file, dygraph_mode_guard


class Task(metaclass=abc.ABCMeta):
    """
    The meta classs of task in Taskflow. The meta class has the five abstract function,
        the subclass need to inherit from the meta class.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, model, task, priority_path=None, **kwargs):
        self.model = model
        self.is_static_model = kwargs.get("is_static_model", False)
        self.task = task
        self.kwargs = kwargs
        self._priority_path = priority_path
        self._usage = ""
        # The dygraph model instance
        self._model = None
        # The static model instance
        self._input_spec = None
        self._config = None
        self._init_class = None
        self._custom_model = False
        self._param_updated = False

        self._num_threads = self.kwargs["num_threads"] if "num_threads" in self.kwargs else math.ceil(cpu_count() / 2)
        self._infer_precision = self.kwargs["precision"] if "precision" in self.kwargs else "fp32"
        # Default to use Paddle Inference
        self._predictor_type = "paddle-inference"
        # The root directory for storing Taskflow related files, default to ~/.paddlenlp.
        self._home_path = self.kwargs["home_path"] if "home_path" in self.kwargs else PPNLP_HOME
        self._task_flag = self.kwargs["task_flag"] if "task_flag" in self.kwargs else self.model
        self.from_hf_hub = kwargs.pop("from_hf_hub", False)

        if "task_path" in self.kwargs:
            self._task_path = self.kwargs["task_path"]
            self._custom_model = True
        elif self._priority_path:
            self._task_path = os.path.join(self._home_path, "taskflow", self._priority_path)
        else:
            self._task_path = os.path.join(self._home_path, "taskflow", self.task, self.model)

        if not self.from_hf_hub:
            download_check(self._task_flag)

    @abstractmethod
    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

    @abstractmethod
    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """

    @abstractmethod
    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """

    @abstractmethod
    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """

    @abstractmethod
    def _postprocess(self, inputs):
        """
        The model output is the logits and pros, this function will convert the model output to raw text.
        """

    @abstractmethod
    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """

    def _check_task_files(self):
        """
        Check files required by the task.
        """
        for file_id, file_name in self.resource_files_names.items():
            path = os.path.join(self._task_path, file_name)
            url = self.resource_files_urls[self.model][file_id][0]
            md5 = self.resource_files_urls[self.model][file_id][1]

            downloaded = True
            if not os.path.exists(path):
                downloaded = False
            else:
                if not self._custom_model:
                    if os.path.exists(path):
                        # Check whether the file is updated
                        if not md5file(path) == md5:
                            downloaded = False
                            if file_id == "model_state":
                                self._param_updated = True
                    else:
                        downloaded = False
            if not downloaded:
                download_file(self._task_path, file_name, url, md5)

    def _check_predictor_type(self):
        if paddle.get_device() == "cpu" and self._infer_precision == "fp16":
            logger.warning("The inference precision is change to 'fp32', 'fp16' inference only takes effect on gpu.")
        else:
            if self._infer_precision == "fp16":
                self._predictor_type = "onnxruntime"

    def _construct_ocr_engine(self, lang="ch", use_angle_cls=True):
        """
        Construct the OCR engine
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("Please install the dependencies first, pip install paddleocr")
        use_gpu = False if paddle.get_device() == "cpu" else True
        self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, show_log=False, use_gpu=use_gpu, lang=lang)

    def _construce_layout_analysis_engine(self):
        """
        Construct the layout analysis engine
        """
        try:
            from paddleocr import PPStructure
        except ImportError:
            raise ImportError("Please install the dependencies first, pip install paddleocr")
        self._layout_analysis_engine = PPStructure(table=False, ocr=True, show_log=False)

    def _prepare_static_mode(self):
        """
        Construct the input data and predictor in the PaddlePaddele static mode.
        """
        if paddle.get_device() == "cpu":
            self._config.disable_gpu()
            self._config.enable_mkldnn()
        else:
            self._config.enable_use_gpu(100, self.kwargs["device_id"])
            # TODO(linjieccc): enable embedding_eltwise_layernorm_fuse_pass after fixed
            self._config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
        self._config.set_cpu_math_library_num_threads(self._num_threads)
        self._config.switch_use_feed_fetch_ops(False)
        self._config.disable_glog_info()
        self._config.enable_memory_optim()

        # TODO(linjieccc): some temporary settings and will be remove in future
        # after fixed
        if self.task in ["document_intelligence", "knowledge_mining"]:
            self._config.switch_ir_optim(False)
        if self.model == "uie-data-distill-gp":
            self._config.enable_memory_optim(False)

        self.predictor = paddle.inference.create_predictor(self._config)
        self.input_names = [name for name in self.predictor.get_input_names()]
        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handle = [self.predictor.get_output_handle(name) for name in self.predictor.get_output_names()]

    def _prepare_onnx_mode(self):
        try:
            import onnx
            import onnxruntime as ort
            import paddle2onnx
            from onnxconverter_common import float16
        except ImportError:
            logger.warning(
                "The inference precision is change to 'fp32', please install the dependencies that required for 'fp16' inference, pip install onnxruntime-gpu onnx onnxconverter-common"
            )

        onnx_dir = os.path.join(self._task_path, "onnx")
        if not os.path.exists(onnx_dir):
            os.mkdir(onnx_dir)
        float_onnx_file = os.path.join(onnx_dir, "model.onnx")
        if not os.path.exists(float_onnx_file) or self._param_updated:
            onnx_model = paddle2onnx.command.c_paddle_to_onnx(
                model_file=self._static_model_file,
                params_file=self._static_params_file,
                opset_version=13,
                enable_onnx_checker=True,
            )
            with open(float_onnx_file, "wb") as f:
                f.write(onnx_model)
        fp16_model_file = os.path.join(onnx_dir, "fp16_model.onnx")
        if not os.path.exists(fp16_model_file) or self._param_updated:
            onnx_model = onnx.load_model(float_onnx_file)
            trans_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
            onnx.save_model(trans_model, fp16_model_file)
        providers = [("CUDAExecutionProvider", {"device_id": self.kwargs["device_id"]})]
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._num_threads
        sess_options.inter_op_num_threads = self._num_threads
        self.predictor = ort.InferenceSession(fp16_model_file, sess_options=sess_options, providers=providers)
        assert "CUDAExecutionProvider" in self.predictor.get_providers(), (
            "The environment for GPU inference is not set properly. "
            "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. "
            "Please run the following commands to reinstall: \n "
            "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"
        )

    def _get_inference_model(self):
        """
        Return the inference program, inputs and outputs in static mode.
        """
        if self._custom_model:
            param_path = os.path.join(self._task_path, "model_state.pdparams")

            if os.path.exists(param_path):
                cache_info_path = os.path.join(self._task_path, ".cache_info")
                md5 = md5file(param_path)
                self._param_updated = True
                if os.path.exists(cache_info_path) and open(cache_info_path).read()[:-8] == md5:
                    self._param_updated = False
                elif self.task == "information_extraction" and self.model != "uie-data-distill-gp":
                    # UIE related models are moved to paddlenlp.transformers after v2.4.5
                    # So we convert the parameter key names for compatibility
                    # This check will be discard in future
                    fp = open(cache_info_path, "w")
                    fp.write(md5 + "taskflow")
                    fp.close()
                    model_state = paddle.load(param_path)
                    prefix_map = {"UIE": "ernie", "UIEM": "ernie_m", "UIEX": "ernie_layout"}
                    new_state_dict = {}
                    for name, param in model_state.items():
                        if "ernie" in name:
                            new_state_dict[name] = param
                        elif "encoder.encoder" in name:
                            trans_name = name.replace("encoder.encoder", prefix_map[self._init_class] + ".encoder")
                            new_state_dict[trans_name] = param
                        elif "encoder" in name:
                            trans_name = name.replace("encoder", prefix_map[self._init_class])
                            new_state_dict[trans_name] = param
                        else:
                            new_state_dict[name] = param
                    paddle.save(new_state_dict, param_path)
                else:
                    fp = open(cache_info_path, "w")
                    fp.write(md5 + "taskflow")
                    fp.close()

        # When the user-provided model path is already a static model, skip to_static conversion
        if self.is_static_model:
            self.inference_model_path = self._task_path
        else:
            # Since 'self._task_path' is used to load the HF Hub path when 'from_hf_hub=True', we construct the static model path in a different way
            _base_path = (
                self._task_path
                if not self.from_hf_hub
                else os.path.join(self._home_path, "taskflow", self.task, self.model)
            )
            self.inference_model_path = os.path.join(_base_path, "static", "inference")
        if not os.path.exists(self.inference_model_path + ".pdiparams") or self._param_updated:
            with dygraph_mode_guard():
                self._construct_model(self.model)
                self._construct_input_spec()
                self._convert_dygraph_to_static()

        self._static_model_file = self.inference_model_path + ".pdmodel"
        self._static_params_file = self.inference_model_path + ".pdiparams"
        if self._predictor_type == "paddle-inference":
            self._config = paddle.inference.Config(self._static_model_file, self._static_params_file)
            self._prepare_static_mode()
        else:
            self._prepare_onnx_mode()

    def _convert_dygraph_to_static(self):
        """
        Convert the dygraph model to static model.
        """
        assert (
            self._model is not None
        ), "The dygraph model must be created before converting the dygraph model to static model."
        assert (
            self._input_spec is not None
        ), "The input spec must be created before converting the dygraph model to static model."
        logger.info("Converting to the inference model cost a little time.")
        static_model = paddle.jit.to_static(self._model, input_spec=self._input_spec)

        paddle.jit.save(static_model, self.inference_model_path)
        logger.info("The inference model save in the path:{}".format(self.inference_model_path))

    def _check_input_text(self, inputs):
        """
        Check whether the input text meet the requirement.
        """
        inputs = inputs[0]
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError("Invalid inputs, input text should not be empty text, please check your input.")
            inputs = [inputs]
        elif isinstance(inputs, list):
            if not (isinstance(inputs[0], str) and len(inputs[0].strip()) > 0):
                raise TypeError(
                    "Invalid inputs, input text should be list of str, and first element of list should not be empty text."
                )
        else:
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, but type of {} found!".format(type(inputs))
            )
        return inputs

    def _auto_splitter(self, input_texts, max_text_len, bbox_list=None, split_sentence=False):
        """
        Split the raw texts automatically for model inference.
        Args:
            input_texts (List[str]): input raw texts.
            max_text_len (int): cutting length.
            bbox_list (List[float, float,float, float]): bbox for document input.
            split_sentence (bool): If True, sentence-level split will be performed.
                `split_sentence` will be set to False if bbox_list is not None since sentence-level split is not support for document.
        return:
            short_input_texts (List[str]): the short input texts for model inference.
            input_mapping (dict): mapping between raw text and short input texts.
        """
        input_mapping = {}
        short_input_texts = []
        cnt_org = 0
        cnt_short = 0
        with_bbox = False
        if bbox_list:
            with_bbox = True
            short_bbox_list = []
            if split_sentence:
                logger.warning(
                    "`split_sentence` will be set to False if bbox_list is not None since sentence-level split is not support for document."
                )
                split_sentence = False

        for idx in range(len(input_texts)):
            if not split_sentence:
                sens = [input_texts[idx]]
            else:
                sens = cut_chinese_sent(input_texts[idx])
            for sen in sens:
                lens = len(sen)
                if lens <= max_text_len:
                    short_input_texts.append(sen)
                    if with_bbox:
                        short_bbox_list.append(bbox_list[idx])
                    input_mapping.setdefault(cnt_org, []).append(cnt_short)
                    cnt_short += 1
                else:
                    temp_text_list = [sen[i : i + max_text_len] for i in range(0, lens, max_text_len)]
                    short_input_texts.extend(temp_text_list)
                    if with_bbox:
                        temp_bbox_list = [bbox_list[idx][i : i + max_text_len] for i in range(0, lens, max_text_len)]
                        short_bbox_list.extend(temp_bbox_list)
                    short_idx = cnt_short
                    cnt_short += math.ceil(lens / max_text_len)
                    temp_text_id = [short_idx + i for i in range(cnt_short - short_idx)]
                    input_mapping.setdefault(cnt_org, []).extend(temp_text_id)
            cnt_org += 1
        if with_bbox:
            return short_input_texts, short_bbox_list, input_mapping
        else:
            return short_input_texts, input_mapping

    def _auto_joiner(self, short_results, input_mapping, is_dict=False):
        """
        Join the short results automatically and generate the final results to match with the user inputs.
        Args:
            short_results (List[dict] / List[List[str]] / List[str]): input raw texts.
            input_mapping (dict): cutting length.
            is_dict (bool): whether the element type is dict, default to False.
        return:
            short_input_texts (List[str]): the short input texts for model inference.
        """
        concat_results = []
        elem_type = {} if is_dict else []
        for k, vs in input_mapping.items():
            single_results = elem_type
            for v in vs:
                if len(single_results) == 0:
                    single_results = short_results[v]
                elif isinstance(elem_type, list):
                    single_results.extend(short_results[v])
                elif isinstance(elem_type, dict):
                    for sk in single_results.keys():
                        if isinstance(single_results[sk], str):
                            single_results[sk] += short_results[v][sk]
                        else:
                            single_results[sk].extend(short_results[v][sk])
                else:
                    raise ValueError(
                        "Invalid element type, the type of results "
                        "for each element should be list of dict, "
                        "but {} received.".format(type(single_results))
                    )
            concat_results.append(single_results)
        return concat_results

    def help(self):
        """
        Return the usage message of the current task.
        """
        print("Examples:\n{}".format(self._usage))

    def __call__(self, *args):
        inputs = self._preprocess(*args)
        outputs = self._run_model(inputs)
        results = self._postprocess(outputs)
        return results

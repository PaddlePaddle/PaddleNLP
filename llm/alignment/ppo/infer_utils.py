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

from __future__ import annotations

import copy
import inspect
import types
from contextlib import contextmanager

import paddle
import paddle.distributed as dist
from comm_utils import cleanup_tensor_space, offload_tensor_to_cpu, reload_tensor_to_gpu
from paddle.utils import try_import
from trainer_utils import guard_set_args

import paddlenlp
from paddlenlp.trainer.trainer import Trainer, logger
from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer
from paddlenlp.transformers.model_utils import dtype_guard


class Predictor:
    def __init__(self, config, model: PretrainedModel = None, tokenizer: PretrainedTokenizer = None):
        self.model_config = model.config
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.is_available = False
        self._weights_mapping = None
        # TODO(guosheng): Removde dependency on llm.Predictor
        # 1. buffer_maker creates caches and other buffer inputs can be shared
        # among multi time prediction. define caches and extra inputs creation
        # method instead of using predictor.__init__
        # 2. inputs_processer creates caches and other inputs can be shared among
        # multi time prediction. define caches and extra inputs creation method
        # instead of using predictor.__init__
        from predictor import InferencePredictorMixin

        self._buffer_maker = types.MethodType(InferencePredictorMixin.__init__, self)
        self._inputs_processer = types.MethodType(InferencePredictorMixin._preprocess, self)

    @staticmethod
    def create_predictor(trainer):
        from predictor import PdArgumentParser, PredictorArgument

        from paddlenlp.trl.llm_utils import get_model_max_position_embeddings

        # create infer model
        # NOTE: infer model use static name param_attr to create and cannot be
        # created multiple times.
        def create_infer_model(model, dtype, set_state=False):
            from models.infer_model_utils import patch_infer_generate

            # apply patches to make FuseMT adapt
            patch_infer_generate(
                eos_token_id=trainer.tokenizer.eos_token_id, pad_token_id=trainer.tokenizer.pad_token_id
            )
            config = copy.deepcopy(model.config)
            hcg = dist.fleet.get_hybrid_communicate_group()  # may differ with training
            config.tensor_parallel_degree = hcg.get_model_parallel_world_size()
            config.tensor_parallel_rank = hcg.get_model_parallel_rank()
            config.quant_type = None
            config.single_card_ptq = True
            infer_model_cls = getattr(paddlenlp.experimental.transformers, model.__class__.__name__ + "InferenceModel")
            # ori_init_weights = infer_model_cls.init_weights
            # infer_model_cls.init_weights = lambda self: None
            with dtype_guard(dtype):
                infer_model = infer_model_cls(config)
            # infer_model_cls.init_weights = ori_init_weights

            if set_state:
                state_dict = {}
                for k, v in model.state_dict().items():
                    # state_dict[k] = np.from_dlpack(paddle.utils.dlpack.to_dlpack(v))
                    state_dict[k] = v.numpy()
                infer_model.set_state_dict(state_dict)
            return infer_model

        # to avoid oom, clear param of infer_model imediately
        ori_creat_param = paddle.nn.Layer.create_parameter

        def _create_param(self, *args, **kwargs):
            param = ori_creat_param(self, *args, **kwargs)
            param._clear_data()
            # param._clear()
            return param

        paddle.nn.Layer.create_parameter = _create_param
        # trainer might use an extra model instead of trainer.model for eval
        eval_model = getattr(trainer, "_inner_eval_model", None)
        infer_model = create_infer_model(trainer.model if eval_model is None else eval_model, dtype=trainer.amp_dtype)
        paddle.nn.Layer.create_parameter = ori_creat_param
        # for k, v in infer_model.state_dict().items():
        #     v._clear()

        # create predictor
        parser = PdArgumentParser((PredictorArgument,))
        predictor_args = parser.parse_dict(
            {
                "src_length": get_model_max_position_embeddings(  # can be changed dynamically by predictor.input_length
                    trainer.model.config if eval_model is None else eval_model.config
                ),
                "max_length": trainer.args.max_length,
                "dtype": trainer.amp_dtype,
                "batch_size": trainer.args.per_device_train_batch_size,
                # infer model do not support top_k, and differ with non-infer model
                # generation which gets default top_K=50 using generation_config.top_k
                "top_p": trainer.args.top_p,
                "temperature": trainer.args.temperature,
                "repetition_penalty": trainer.args.repetition_penalty,
            }
        )[0]
        policy_predictor = Predictor(predictor_args, model=infer_model, tokenizer=trainer.tokenizer)
        return policy_predictor

    def _create_caches(self):
        """inputs can be reused among multiple predictions, such as cache"""
        if hasattr(self, "cache_kvs_shape"):  # has created cache
            input_length = getattr(self, "input_length", 0)
            # TODO(guosheng): better way to get history max cahce length, we can
            # not get cahce length form cache tensor when not know cache layout
            if input_length <= self.config.src_length:  # reuse cahce
                return
            else:  # create longer cache
                self._clear_caches()
        self.config.src_length = getattr(self, "input_length", self.config.src_length)
        if not hasattr(self, "_buffer_attrs"):
            pre_attrs = set(self.__dict__.keys())
        self.cache_kvs_shape = self.model.get_cache_kvs_shape(
            self.model_config, self.config.batch_size, self.config.total_max_length
        )
        self._buffer_maker(self.config, self.tokenizer)
        if not hasattr(self, "_buffer_attrs"):
            self._buffer_attrs = set(self.__dict__.keys()) - pre_attrs

    def _clear_caches(self):
        # del or offload
        for attr in self._buffer_attrs:
            delattr(self, attr)

    def disable(self, model, onload_model=True):
        # clear caches
        self._clear_caches()
        # clear params
        for _, param in self.model.state_dict().items():
            param._clear_data()
            # param._clear()
        if onload_model:
            model.to(paddle.device.get_device())
        self.is_available = False

    def enable(self, model, offload_model=True):
        if self.is_available:
            return
        # set params
        self.set_state_dict(model, offload_model)
        self.is_available = True

    @paddle.no_grad()
    def set_state_dict(self, model, offload_model=True):
        offload_place = paddle.CUDAPinnedPlace()
        state_dict = {}
        for k, v in model.state_dict().items():
            state_dict[k] = v

        if getattr(self, "_weights_mapping", None) is None:
            self._weights_mapping = self.model.get_weights_mapping()

        # non_share_params = []
        for k, v in self._weights_mapping.items():
            param, (convert_fun, args) = k, v
            args = [state_dict[name] for name in args]
            value = convert_fun(*args)
            if offload_model:
                for arg in args:
                    # shared params no need to offload
                    if value is not arg:
                        cpu_arg = arg._copy_to(offload_place, blocking=False)
                        cpu_arg._share_buffer_to(arg)
            if not isinstance(value, paddle.Tensor):
                param.set_value(value)
            # elif isinstance(value.place, paddle.CUDAPlace):
            elif value.place.is_gpu_place():
                # NOTE: _share_buffer_to seems do not work
                # value._share_buffer_to(param)
                # value._share_underline_tensor_to(param)
                param.get_tensor()._share_data_with(value.get_tensor())
            else:
                param.copy_(value, True)

        paddle.device.cuda.synchronize()

    def _preprocess(self, source):
        # make cache when infer happens to get actual shape to save memory
        self._create_caches()
        with guard_set_args(self.config, {"src_length": getattr(self, "input_length", self.config.src_length)}):
            inputs = self._inputs_processer(source)
        # We want to use a defined input_length to create cache and input_ids.
        # However predictor could not use a specified length to pad currently.
        # Thus we use this way to let get the actual input length.
        self.infer_input_length = inputs["input_ids"].shape[-1]
        return inputs

    @paddle.no_grad()
    def _infer(self, inputs):
        for key in inputs.keys():
            if paddle.is_tensor(inputs[key]):
                continue
            if isinstance(inputs[key], list):
                if paddle.is_tensor(inputs[key]):
                    continue
                inputs[key] = [paddle.to_tensor(item) for item in inputs[key]]
            else:
                inputs[key] = paddle.to_tensor(inputs[key])

        inputs["cache_kvs"] = self.cache_kvs
        return self.model.generate(**inputs)

    def _postprocess(self, predictions):
        return predictions

    @paddle.no_grad()
    def predict(self, input_texts: str | list[str]):
        tokenized_source = self._preprocess(input_texts)
        predictions = self._infer(tokenized_source)
        decoded_predictions = self._postprocess(predictions)
        return decoded_predictions


policy_predictor: Predictor = None


@contextmanager
def infer_guard(trainer, offload_model=True):
    # trainer might use an extra model instead of trainer.model for eval
    eval_model = getattr(trainer, "_inner_eval_model", None)
    model = trainer.model if eval_model is None else eval_model

    # PipelineParallel does not support inference speedup
    if not getattr(trainer, "use_fusemt", False) or isinstance(
        model, (dist.fleet.meta_parallel.PipelineLayer, dist.fleet.model.PipelineParallel)
    ):
        yield
        return

    try:
        try_import("paddlenlp_ops")
    except:
        logger.warning("paddlenlp_ops does not exist, please install paddlenlp_ops for generation speedup.")
        yield
        return

    global policy_predictor
    if policy_predictor is None:
        policy_predictor = Predictor.create_predictor(trainer)
    if not policy_predictor.is_available:
        policy_predictor.enable(model, offload_model=offload_model)

    # TODO(guosheng): patch for dist.all_recude to use tp group, fix it later
    ori_all_reduce = dist.all_reduce
    ori_broadcast = dist.broadcast
    hcg = dist.fleet.get_hybrid_communicate_group()
    dist.all_reduce = lambda x: ori_all_reduce(x, group=hcg.get_model_parallel_group())
    dist.broadcast = lambda x, rank: ori_broadcast(
        x, src=hcg.get_model_parallel_group_src_rank(), group=hcg.get_model_parallel_group()
    )
    yield
    dist.all_reduce = ori_all_reduce
    dist.broadcast = ori_broadcast

    policy_predictor.disable(model, onload_model=offload_model)


class InferEvalModel:
    """For faster generation, not support PipelineParallel yet."""

    def __init__(self, trainer: Trainer):
        # trainer might use an extra model instead of trainer.model for eval
        eval_model = getattr(trainer, "_inner_eval_model", None)
        self.model: PretrainedModel = trainer.model if eval_model is None else eval_model
        self.tokenizer: PretrainedTokenizer = trainer.tokenizer
        self.trainer = trainer

    def enable(self):
        trainer = self.trainer
        if trainer.model is not self.model:
            trainer.export_evaluate_model(
                trainer.model,
                self.model,
                with_offload="train_model" in trainer.args.offload_level,
            )
        else:
            reload_tensor_to_gpu(self.model.state_dict())

    def disable(self):
        trainer = self.trainer
        if trainer.model is not self.model:
            cleanup_tensor_space(self.model.state_dict())
        else:
            offload_tensor_to_cpu(self.model.state_dict())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def __call__(self, *args, **kwargs):
        # assert model is on GPU
        assert policy_predictor is None or not policy_predictor.is_available
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if policy_predictor is None or not policy_predictor.is_available:
            return self.model.generate(*args, **kwargs)

        arg_dict = inspect.signature(self.model.generate).bind(*args, **kwargs).arguments
        input_ids = arg_dict["input_ids"]
        generation_config = arg_dict["generation_config"]
        # convert text and tokenize again to convert left padding to right padding
        # remove this if inputs is right padding
        # TODO(guosheng): allow to use right padding to infer directly
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # decoded prompts has been applied with chat_template
        # NOTE(guosheng): Whether to add special token should be checked, None
        # chat_template would not add special token in predictor, since it assumes
        # chat_template includes special tokens. While Beaver dataset tokenization
        # does not use chat_template, it uses hard coded template which excludes
        # special tokens.
        with guard_set_args(
            policy_predictor.tokenizer,
            {
                # predictor use right padding for infer model by default
                # "padding_side": "right",
                # "chat_template": None
            },
        ):
            # NOTE: right padding in predictor according to prompt might have a
            # different length with input_ids, espically when input_ids has more
            # paddings than the necessary. Thus pass input_length to predictor to:
            # 1. use a consistent length to replace input_ids back to output to
            #    keep the same padding format. however predictor could not use a
            #    specified length to pad currently
            # 2. allow to use a dynamic length for memory efficiency (by a smaller
            #    cache)
            policy_predictor.input_length = input_ids.shape[-1]
            outputs = policy_predictor.predict(prompts)

        if generation_config.trunc_input:
            outputs = (outputs[0][:, policy_predictor.infer_input_length :],)
            return outputs

        if policy_predictor.input_length != policy_predictor.infer_input_length:
            outputs = (paddle.concat([input_ids, outputs[0][:, policy_predictor.infer_input_length :]], axis=-1),)
            return outputs

        outputs = (outputs[0],)
        if self.tokenizer.padding_side == "left":
            # convert back to left padding inputs
            outputs[0][:, : input_ids.shape[-1]] = input_ids
        return outputs

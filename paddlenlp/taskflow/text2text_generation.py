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

import paddle

from ..transformers import AutoModelForCausalLM, AutoTokenizer
from ..utils.log import logger
from .task import Task
from .utils import static_mode_guard


class ChatGLMTask(Task):
    """
    The text to text generation LLM model to predict the question or chinese  poetry.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        # Default to static mode
        self._static_mode = False
        self._dtype = kwargs.get("dtype", "float16")
        self.kwargs["generation_task"] = task
        self._tgt_length = kwargs.get("tgt_length", 2048)
        # Token max length
        self._max_seq_length = kwargs.get("max_seq_length", 2048)
        self._top_k = kwargs.get("top_k", 1)
        self._top_p = kwargs.get("top_p", 1.0)
        self._temperature = kwargs.get("temperature", 1.0)
        self._decode_strategy = kwargs.get("decode_strategy", "sampling")
        self._num_return_sequences = kwargs.get("num_return_sequences", 1)

        self._construct_tokenizer(model)
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._construct_input_spec()

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None, None, None], dtype="int64"),  # attention_mask
            paddle.static.InputSpec(shape=[None, None, None], dtype="int64"),  # position_ids
            # max_length
            self._tgt_length,
            # min_length
            0,
            # decode_strategy
            self._decode_strategy,
            # temperature
            self._temperature,
            # top_k
            self._top_k,
            # top_p
            self._top_p,
            # repetition_penalty
            1,
            # num_beams
            1,
            # num_beam_groups
            1,
            # length_penalty
            0.0,
            # early_stopping
            False,
            # bos_token_id
            self._tokenizer.bos_token_id,
            # eos_token_id
            self._tokenizer.eos_token_id,
            # pad_token_id
            self._tokenizer.pad_token_id,
            # decoder_start_token_id
            None,
            # forced_bos_token_id
            None,
            # forced_eos_token_id
            None,
            # no_repeat_ngram_size
            None,
            # num_return_sequences
            self._num_return_sequences,
            # diversity_rate
            0.0,
            # use_cache
            True,
        ]

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        tokenizer_instance = AutoTokenizer.from_pretrained(model)

        self._tokenizer = tokenizer_instance

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = AutoModelForCausalLM.from_pretrained(
            self.model,
            load_state_as_np=True,
            dtype=self._dtype,
        )
        # Load the model parameter for the predict
        model_instance.eval()
        self._model = model_instance

    def _batchify(self, data, batch_size):
        """
        Generate input batches.
        """
        # Separates data into some batches.
        one_batch = []
        for example in data:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield one_batch
                one_batch = []
        if one_batch:
            yield one_batch

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        # Get the config from the kwargs
        batch_size = self.kwargs["batch_size"] if "batch_size" in self.kwargs else 1
        batches = self._batchify(inputs, batch_size)
        examples = []
        for input_text in batches:
            if self._static_mode:
                tokenized_output = self._tokenizer(
                    input_text,
                    return_tensors="np",
                    padding=True,
                    max_length=self._max_seq_length,
                    truncation=True,
                    truncation_side="left",
                )
            else:
                tokenized_output = self._tokenizer(
                    input_text,
                    return_tensors="pd",
                    padding=True,
                    max_length=self._max_seq_length,
                    truncation=True,
                    truncation_side="left",
                )
            examples.append(tokenized_output)
        outputs = {}
        outputs["text"] = inputs
        outputs["data_loader"] = examples
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        results = []
        if self._static_mode:
            with static_mode_guard():
                for batch in inputs["data_loader"]:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    position_ids = batch["position_ids"]
                    self.input_handles[0].copy_from_cpu(input_ids)
                    self.input_handles[1].copy_from_cpu(attention_mask)
                    self.input_handles[2].copy_from_cpu(position_ids)
                    self.predictor.run()
                    result = self.output_handle[0].copy_to_cpu().tolist()
                    results.extend(result)
        else:
            for batch_inputs in inputs["data_loader"]:
                result = self._model.generate(
                    **batch_inputs,
                    decode_strategy=self._decode_strategy,
                    top_k=self._top_k,
                    top_p=self._top_p,
                    temperature=self._temperature,
                    max_length=self._tgt_length,
                    bos_token_id=self._tokenizer.bos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    pad_token_id=self._tokenizer.pad_token_id,
                    num_return_sequences=self._num_return_sequences,
                    use_cache=True,
                )
                result = result[0]
                results.extend(result)

        inputs["results"] = results
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        preds = inputs["results"]
        result = []
        for x in preds:
            if self._static_mode:
                res = self._tokenizer.decode(x, skip_special_tokens=True)
                res = res.strip("\n")
                result.append(res)
            else:
                res = self._tokenizer.decode(x.numpy().tolist(), skip_special_tokens=True)
                res = res.strip("\n")
                result.append(res)
        out_dict = {"result": result}
        return out_dict

    def set_argument(self, argument: dict):
        for k, v in argument.items():
            if k == "input":
                continue
            setattr(self, f"_{k}", v)

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

        static_model = paddle.jit.to_static(self._model.generate, input_spec=self._input_spec)
        paddle.jit.save(static_model, self.inference_model_path)
        logger.info("The inference model save in the path:{}".format(self.inference_model_path))

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest
import os
import numpy as np
import paddle
from parameterized import parameterized_class, parameterized

from paddlenlp.transformers.rw.modeling import (
    RWModel,
    RWForCausalLM,
    RWForQuestionAnswering,
    RWForTokenClassification,
)
from paddlenlp.transformers.rw.configuration import RWConfig
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...testing_utils import require_package, slow

class RWModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        type_sequence_label_size=2,
        use_relative_position=True,
        num_labels=3,
        num_choices=4,
        num_classes=3,
        scope=None,
        multi_query=True,
        bias=False,
        parallel_attn=True,
        output_attentions=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.type_sequence_label_size = type_sequence_label_size
        self.use_relative_position = use_relative_position
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.multi_query = multi_query
        self.bias = bias
        self.parallel_attn = parallel_attn
        self.output_attentions = output_attentions

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent and self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict

    def get_config(self):
        return RWConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            use_relative_position=self.use_relative_position,
            num_class=self.num_classes,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
            multi_query = self.multi_query,
            bias = self.bias,
            parallel_attn = self.parallel_attn,
            output_attentions = self.output_attentions,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RWModel(config)
        model.eval()
        result = model(
            input_ids, attention_mask=input_mask, return_dict=self.parent.return_dict,
        )
        result = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        #self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])
        result = model(input_ids, use_cache=True, output_attentions=True, return_dict=self.parent.return_dict)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        #self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])
        result = model(input_ids, use_cache=True, output_attentions=True, output_hidden_states=True, return_dict=self.parent.return_dict)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        #self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_for_causal_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RWForCausalLM(config)
        model.eval()
        result = model(
            input_ids, attention_mask=input_mask, labels=input_ids, return_dict=self.parent.return_dict,
        )
        print(self.parent.return_dict, "return")
        print(result[1], "lm logit")
        print(result[0], "loss")
        #if self.parent.return_dict:
        self.parent.assertEqual(result[0].shape, [])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RWForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            return_dict=self.parent.return_dict,
        )
        if sequence_labels is not None:
            start_logits, end_logits = result[1], result[2]
        else:
            start_logits, end_logits = result[0], result[1]

        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RWForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            labels=token_labels,
            return_dict=self.parent.return_dict,
        )
        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_classes])

@parameterized_class(
     ("return_dict", "use_labels"),
     [
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
 )
class RWModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = RWModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = True

    all_model_classes = (
        RWModel,
        RWForCausalLM,
        RWForQuestionAnswering,
        RWForTokenClassification,
    )

    def setUp(self):
        self.model_tester = RWModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_model(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

# Please execute this unittest on PaddleNLP/
#TorchConfiguration = ''
#TorchModelPath = ''
#TorchModelCausalPath = ''
#TorchModelTokenPath = ''
#TorchModelQAPath = ''
#PaddleModelPath = ''
#
#Falcon7BPath = ""
#Falcon7BInstructPath = ""
#
#Class2Path = {
#    'RWModel': TorchModelPath,
#    'RWForCausalLM': TorchModelCausalPath,
#    'RWForTokenClassification': TorchModelTokenPath,
#    'RWForQuestionAnswering': TorchModelQAPath,
#}

#class RWCompatibilityTest(unittest.TestCase):
#    #test_model_id = "hf-internal-testing/tiny-random-RobertaModel"
#
#    @classmethod
#    @require_package("transformers", "torch")
#    def setUpClass(cls) -> None:
#        from transformers.configuration_utils import PretrainedConfig as TorchPretrainedConfig
#        from paddlenlp.transformers.configuration_utils import PretrainedConfig as PretrainedConfig
#        def get_model(config, model_class):
#            if isinstance(config, TorchPretrainedConfig) or isinstance(config, PretrainedConfig):
#                return model_class(config)
#            if model_class == self.base_model_class:
#                return model_class(**config)
#            return model_class(self.base_model_class(**config))
#
#        if not os.path.isfile(TorchModelPath + 'pytorch_model.bin'):
#            from .torch_modelling_RW import RWModel as TorchRWModel
#            from .torch_config_RW import RWConfig as TorchRWConfig
#
#            if not os.path.exists(TorchModelPath):
#                os.makedirs(TorchModelPath)
#
#            config = TorchRWConfig.from_pretrained(TorchConfiguration)
#            tiny_model = get_model(config, TorchRWModel)
#
#            tiny_model.save_pretrained(TorchModelPath)
#
#        if not os.path.isfile(TorchModelCausalPath + 'pytorch_model.bin'):
#            from .torch_modelling_RW import RWForCausalLM as TorchRWCausalLM
#            from .torch_config_RW import RWConfig as TorchRWConfig
#
#            if not os.path.exists(TorchModelCausalPath):
#                os.makedirs(TorchModelCausalPath)
#
#            config = TorchRWConfig.from_pretrained(TorchConfiguration)
#            tiny_model = get_model(config, TorchRWCausalLM)
#
#            tiny_model.save_pretrained(TorchModelCausalPath)
#
#        if not os.path.isfile(TorchModelTokenPath + 'pytorch_model.bin'):
#            from .torch_modelling_RW import RWForTokenClassification as TorchRWTokenClassification
#            from .torch_config_RW import RWConfig as TorchRWConfig
#
#            if not os.path.exists(TorchModelTokenPath):
#                os.makedirs(TorchModelTokenPath)
#
#            config = TorchRWConfig.from_pretrained(TorchConfiguration)
#            tiny_model = get_model(config, TorchRWTokenClassification)
#
#            tiny_model.save_pretrained(TorchModelTokenPath)
#
#        if not os.path.isfile(TorchModelQAPath + 'pytorch_model.bin'):
#            from .torch_modelling_RW import RWForQuestionAnswering as TorchRWQA
#            from .torch_config_RW import RWConfig as TorchRWConfig
#
#            if not os.path.exists(TorchModelQAPath):
#                os.makedirs(TorchModelQAPath)
#
#            config = TorchRWConfig.from_pretrained(TorchConfiguration)
#            tiny_model = get_model(config, TorchRWQA)
#
#            tiny_model.save_pretrained(TorchModelQAPath)
#
#        if not os.path.isfile(PaddleModelPath + 'model_state.pdparams'):
#            from paddlenlp.transformers.rw.configuration_RW import RWConfig
#            from paddlenlp.transformers.rw.modelling import RWForQuestionAnswering
#
#            if not os.path.exists(TorchModelPath):
#                os.makedirs(TorchModelPath)
#
#            config = RWConfig.from_pretrained(TorchConfiguration)
#            tiny_model = get_model(config, RWForQuestionAnswering)
#            import pdb; pdb.set_trace()
#            tiny_model.save_pretrained(PaddleModelPath)
#
#    @parameterized.expand(
#        [
#            ("RWForCausalLM", ),  #TODO: need to tie weights
#            ("RWModel",),
#            ("RWForTokenClassification",),
#            ("RWForQuestionAnswering",),
#        ]
#    )
#    @require_package("transformers", "torch")
#    def utest_rw_classes_from_tiny_model(self, class_name):
#        # 1. create commmon input
#        input_ids = np.random.randint(3, 100, [5, 20])
#
#        # 2. forward the paddle model
#        from paddlenlp.transformers.rw import modelling
#
#        paddle_model_class = getattr(modelling, class_name)
#        paddle_model = paddle_model_class.from_pretrained(Class2Path[class_name], convert_from_torch=True)
#        paddle_model.eval()
#        #paddle_outputs = paddle_model(paddle.to_tensor(input_ids), return_dict=False, output_hidden_states=True)
#        paddle_outputs = paddle_model(paddle.to_tensor(input_ids), return_dict=False)
#        paddle_logits = paddle_outputs[0]
#        #paddle_hd = paddle_outputs[2]
#
#        # 3. forward the torch model
#        import torch
#        from . import torch_modelling_RW
#        torch_model_class = getattr(torch_modelling_RW, class_name)
#
#        torch_model = torch_model_class.from_pretrained(Class2Path[class_name])
#        torch_model.eval()
#        #torch_outputs = torch_model(torch.tensor(input_ids), return_dict=False, output_hidden_states=True)
#        torch_outputs = torch_model(torch.tensor(input_ids), return_dict=False)
#        torch_logits = torch_outputs[0]
#        #torch_hd = torch_outputs[2]
#
#        self.assertTrue(
#            np.allclose(
#                paddle_logits.detach().cpu().reshape([-1])[:9].numpy(),
#                torch_logits.detach().cpu().reshape([-1])[:9].numpy(),
#                atol=1e-3,
#            )
#        )
#
#    @parameterized.expand(
#        [
#            ("RWForCausalLM",),  #TODO: need to tie weights
#            #("RWModel",),
#            #("RWForTokenClassification",),
#            #("RWForQuestionAnswering",),
#        ]
#    )
#    def test_rw_classes_from_local_dir(self, class_name):
#        # 1. create commmon input
#        #input_ids = np.random.randint(1, 100, [5, 20])
#        input_ids = np.load('./tests/transformers/rw/input_ids.npy')
#
#        # 3. forward the torch model
#        import torch
#        from . import torch_modelling_RW
#        torch_model_class = getattr(torch_modelling_RW, class_name)
#
#        torch_model = torch_model_class.from_pretrained(Falcon7BPath)
#        torch_model.eval()
#        #torch_outputs = torch_model(torch.tensor(input_ids), return_dict=False, output_hidden_states=True)
#        torch_outputs = torch_model(torch.tensor(input_ids), return_dict=False)
#        torch_logits = torch_outputs[0]
#        #torch_hd = torch_outputs[2]
#
#        # 2. forward the paddle model
#        from paddlenlp.transformers.rw import modelling
#
#        paddle_model_class = getattr(modelling, class_name)
#        paddle_model = paddle_model_class.from_pretrained(Falcon7BPath, convert_from_torch=True)
#        paddle_model.eval()
#        #paddle_outputs = paddle_model(paddle.to_tensor(input_ids), return_dict=False, output_hidden_states=True)
#        paddle_outputs = paddle_model(paddle.to_tensor(input_ids), return_dict=False)
#        paddle_logits = paddle_outputs[0]
#        #paddle_hd = paddle_outputs[2]
#        import pdb; pdb.set_trace()
#        #import numpy; numpy.save('./tests/transformers/rw/torch_logit.npy', torch_logits.numpy())
#
#        self.assertTrue(
#            np.allclose(
#                paddle_logits.detach().cpu().numpy(),
#                torch_logits.detach().cpu().numpy(),
#                atol=1e-3,
#            )
#        )

#if __name__ == "__main__":
#    unittest.main()
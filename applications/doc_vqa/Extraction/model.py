import numpy as np

import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddlenlp.transformers import LayoutXLMPretrainedModel


class Crf_decoding(paddle.fluid.dygraph.Layer):

    def __init__(self, param_attr, size=None, is_test=True, dtype='float32'):
        super(Crf_decoding, self).__init__()

        self._dtype = dtype
        self._size = size
        self._is_test = is_test
        self._param_attr = param_attr
        self._transition = self.create_parameter(
            attr=self._param_attr,
            shape=[self._size + 2, self._size],
            dtype=self._dtype)

    @property
    def weight(self):
        return self._transition

    @weight.setter
    def weight(self, value):
        self._transition = value

    def forward(self, input, label=None, length=None):

        viterbi_path = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        this_inputs = {
            "Emission": [input],
            "Transition": self._transition,
            "Label": label
        }
        if length is not None:
            this_inputs['Length'] = [length]
        self._helper.append_op(type='crf_decoding',
                               inputs=this_inputs,
                               outputs={"ViterbiPath": [viterbi_path]},
                               attrs={
                                   "is_test": self._is_test,
                               })
        return viterbi_path


class Chunk_eval(paddle.fluid.dygraph.Layer):

    def __init__(self,
                 num_chunk_types,
                 chunk_scheme,
                 excluded_chunk_types=None):
        super(Chunk_eval, self).__init__()
        self.num_chunk_types = num_chunk_types
        self.chunk_scheme = chunk_scheme
        self.excluded_chunk_types = excluded_chunk_types

    def forward(self, input, label, seq_length=None):

        precision = self._helper.create_variable_for_type_inference(
            dtype="float32")
        recall = self._helper.create_variable_for_type_inference(
            dtype="float32")
        f1_score = self._helper.create_variable_for_type_inference(
            dtype="float32")
        num_infer_chunks = self._helper.create_variable_for_type_inference(
            dtype="int64")
        num_label_chunks = self._helper.create_variable_for_type_inference(
            dtype="int64")
        num_correct_chunks = self._helper.create_variable_for_type_inference(
            dtype="int64")

        this_input = {"Inference": [input], "Label": [label]}
        if seq_length is not None:
            this_input["SeqLength"] = [seq_length]

        self._helper.append_op(type='chunk_eval',
                               inputs=this_input,
                               outputs={
                                   "Precision": [precision],
                                   "Recall": [recall],
                                   "F1-Score": [f1_score],
                                   "NumInferChunks": [num_infer_chunks],
                                   "NumLabelChunks": [num_label_chunks],
                                   "NumCorrectChunks": [num_correct_chunks]
                               },
                               attrs={
                                   "num_chunk_types":
                                   self.num_chunk_types,
                                   "chunk_scheme":
                                   self.chunk_scheme,
                                   "excluded_chunk_types":
                                   self.excluded_chunk_types or []
                               })
        return (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
                num_correct_chunks)


class Linear_chain_crf(paddle.fluid.dygraph.Layer):

    def __init__(self, param_attr, size=None, is_test=False, dtype='float32'):
        super(Linear_chain_crf, self).__init__()

        self._param_attr = param_attr
        self._dtype = dtype
        self._size = size
        self._is_test = is_test
        self._transition = self.create_parameter(
            attr=self._param_attr,
            shape=[self._size + 2, self._size],
            dtype=self._dtype)

    @property
    def weight(self):
        return self._transition

    @weight.setter
    def weight(self, value):
        self._transition = value

    def forward(self, input, label, length=None):

        alpha = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        emission_exps = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        transition_exps = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        log_likelihood = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        this_inputs = {
            "Emission": [input],
            "Transition": self._transition,
            "Label": [label]
        }
        if length is not None:
            this_inputs['Length'] = [length]
        self._helper.append_op(type='linear_chain_crf',
                               inputs=this_inputs,
                               outputs={
                                   "Alpha": [alpha],
                                   "EmissionExps": [emission_exps],
                                   "TransitionExps": transition_exps,
                                   "LogLikelihood": log_likelihood
                               },
                               attrs={
                                   "is_test": self._is_test,
                               })
        return log_likelihood


class LayoutXLMForTokenClassification_with_CRF(LayoutXLMPretrainedModel):

    def __init__(self, layoutxlm, num_classes, dropout=None):
        super(LayoutXLMForTokenClassification_with_CRF, self).__init__()
        self.num_classes = num_classes
        if isinstance(layoutxlm, dict):
            self.layoutxlm = LayoutXLMModel(**layoutxlm)
        else:
            self.layoutxlm = layoutxlm
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  layoutxlm.config["hidden_dropout_prob"])
        self.emission_classifier = nn.Linear(
            self.layoutxlm.config["hidden_size"], self.num_classes)
        self.emission_classifier.apply(self.init_weights)
        self.linear_chain_crf = Linear_chain_crf(
            size=self.num_classes,
            param_attr=paddle.fluid.ParamAttr(name='liner_chain_crfw'))
        self.crf_decoding = Crf_decoding(
            param_attr=paddle.fluid.ParamAttr(name='crfw_decode'),
            size=self.num_classes)
        self.crf_decoding.weight = self.linear_chain_crf.weight
        self.crfw = fluid.layers.create_parameter(
            shape=[self.num_classes + 2, self.num_classes],
            dtype='float32',
            name='crfw')
        self.mask_crfw = fluid.layers.create_parameter(
            shape=[self.num_classes + 2, self.num_classes],
            dtype='float32',
            name='mask_matrix')

    def get_input_embeddings(self):
        return self.layoutxlm.embeddings.word_embeddings

    def forward(self,
                input_ids=None,
                bbox=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                image=None,
                position_ids=None,
                head_mask=None,
                is_train=False):

        input_ids = input_ids.squeeze(axis=1)
        bbox = bbox.squeeze(axis=1)
        attention_mask = attention_mask.squeeze(axis=1)
        token_type_ids = token_type_ids.squeeze(axis=1)
        outputs = self.layoutxlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.shape[1]
        # sequence out and image out
        sequence_logits, image_output = outputs[0][:, :seq_length], outputs[
            0][:, seq_length:]
        emission = self.emission_classifier(sequence_logits)
        length = paddle.sum(attention_mask, axis=1)
        labels = labels.reshape([-1, seq_length, 1])

        # standard crf loss
        crf_cost = self.linear_chain_crf(input=emission,
                                         label=labels,
                                         length=length)
        crf_decode = self.crf_decoding(input=emission, length=length)
        if is_train:
            return [crf_cost]
        else:
            return [crf_cost, crf_decode]

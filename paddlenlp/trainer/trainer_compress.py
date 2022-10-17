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

import time
import os
import copy
import math
import numpy as np
import inspect

import paddle
from paddle.utils import try_import
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization

from ..utils.log import logger
from ..data import Pad
from ..transformers import AutoModelForSequenceClassification
from ..transformers import AutoModelForQuestionAnswering
from ..transformers import AutoModelForTokenClassification
from ..transformers import export_model
from ..transformers.ofa_utils import *
from ..transformers.model_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from ..metrics import ChunkEvaluator
from ..metrics.squad import squad_evaluate, compute_prediction
from .trainer_base import Trainer


def global_try_import_slim():
    global paddleslim
    try_import('paddleslim')
    import paddleslim


def compress(self, custom_evaluate=None):
    """
    Supports pruning DynaBERT and post-training quantization. If both are
    needed, pruning DynaBERT would be performed before quantizaton.
    """
    args = self.args
    self.custom_evaluate = custom_evaluate
    if "dynabert" in args.strategy:
        global_try_import_slim()
        if self.args.width_mult_list is not None:
            self.args.width_mult_list = [
                eval(width_mult) for width_mult in self.args.width_mult_list
            ]
        class_name = self.model.__class__.__name__
        if "SequenceClassification" not in class_name and "TokenClassification" not in class_name and "QuestionAnswering" not in class_name:
            assert self.custom_evaluate is not None, \
                "Custom model using DynaBERT strategy needs to pass in parameters `custom_evaluate`."
        _dynabert(self, self.model, args.output_dir)
        if "ptq" in args.strategy:
            self.args.input_filename_prefix = "pruned_model"
            for width_mult in args.width_mult_list:
                output_dir_width = os.path.join(
                    args.output_dir, "width_mult_" + str(round(width_mult, 2)))
                self.quant(output_dir_width, "ptq")
    elif args.strategy == "ptq":
        # Input model is an inference model
        if args.input_infer_model_path is not None:
            model_dir = os.path.dirname(args.input_infer_model_path)
            self.args.input_filename_prefix = os.path.basename(
                args.input_infer_model_path)
            self.quant(model_dir, args.strategy)
        # Input model is load from Trainer API in dygraph.
        else:
            # Prefix of `export_model` is 'model'
            self.args.input_filename_prefix = "model"
            input_spec = generate_input_spec(self.model, self.train_dataset)
            input_dir = args.output_dir
            export_model(model=self.model,
                         input_spec=input_spec,
                         path=input_dir)
            self.quant(input_dir, args.strategy)
    elif args.strategy == "qat":
        global_try_import_slim()
        self.quant(args.output_dir, args.strategy)


def quant(self, model_dir, strategy):
    """
    Supports Post-Training Quantization now.
    """
    if strategy == "ptq":
        _post_training_quantization_grid_search(self, model_dir)
    elif strategy == 'qat':
        _quant_aware_training_dynamic(self, model_dir)


def generate_input_spec(model, dataset):
    model_para_keys = inspect.signature(model.forward).parameters.keys()
    input_num = 0
    for key in dataset[0].keys():
        if key in model_para_keys and key not in ('labels', 'start_positions',
                                                  'end_positions'):
            input_num += 1
    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype='int64')
        for i in range(input_num)
    ]
    return input_spec


def _dynabert(self, model, output_dir):
    args = self.args
    model = _replace_auto_model_forward(model)
    if args.width_mult_list is None:
        args.width_mult_list = [0.75]
    # Each batch is a dict.
    train_dataloader = self.get_train_dataloader()
    eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
    if "QuestionAnswering" in model.__class__.__name__:
        eval_dataloader_with_label = self.get_eval_dataloader(
            self.eval_examples)
        ofa_model, teacher_model = _dynabert_init(self, model,
                                                  eval_dataloader_with_label)
    else:
        ofa_model, teacher_model = _dynabert_init(self, model, eval_dataloader)

    # TODO: args.gradient_accumulation_steps
    if args.max_steps > 0:
        args.num_training_steps = args.max_steps
        args.num_train_epochs = math.ceil(num_training_steps /
                                          len(train_dataloader))
    else:
        args.num_training_steps = len(train_dataloader) * args.num_train_epochs
        args.num_train_epochs = math.ceil(args.num_train_epochs)
    self.create_optimizer_and_scheduler(
        num_training_steps=args.num_training_steps)

    ofa_model = _dynabert_training(self, ofa_model, model, teacher_model,
                                   train_dataloader, eval_dataloader,
                                   args.num_train_epochs)

    # Each width_mult best model would be exported.
    _dynabert_export(self, ofa_model)

    ofa_model, ofa_model.model = _recover_transformer_func(
        ofa_model, True), _recover_transformer_func(ofa_model.model, True)
    ofa_model.model = _recover_auto_model_forward(ofa_model.model)
    logger.info("Pruning is finished using DynaBERT strategy.")


def _replace_transformer_func(self):
    nn.MultiHeadAttention._ori_forward = paddle.nn.MultiHeadAttention.forward
    nn.MultiHeadAttention._ori_prepare_qkv = nn.MultiHeadAttention._prepare_qkv

    nn.MultiHeadAttention._forward = mha_ofa_forward
    nn.MultiHeadAttention.__prepare_qkv = prepare_qkv_ofa
    nn.TransformerEncoder._forward = encoder_ofa_forward
    nn.TransformerEncoderLayer._forward = encoder_layer_ofa_forward

    def init_func(layer):
        if isinstance(layer, nn.MultiHeadAttention):
            layer.forward = layer._forward
            layer._prepare_qkv = layer.__prepare_qkv
        elif isinstance(layer, nn.TransformerEncoderLayer):
            layer.forward = layer._forward
        elif isinstance(layer, nn.TransformerEncoder):
            layer.forward = layer._forward

    for layer in self.children():
        layer.apply(init_func)
    return self


def _recover_transformer_func(self, all_recover=False):

    def init_func(layer):
        if isinstance(layer, nn.MultiHeadAttention):
            layer.forward = layer._ori_forward
        elif isinstance(layer, nn.TransformerEncoderLayer):
            layer.forward = layer._ori_forward
        elif isinstance(layer, nn.TransformerEncoder):
            layer.forward = layer._ori_forward
        if all_recover:
            if isinstance(layer, nn.MultiHeadAttention):
                layer._prepare_qkv = layer._ori_prepare_qkv

    for layer in self.children():
        layer.apply(init_func)

    return self


def _replace_auto_model_forward(self):
    self.base_model_class._forward = auto_model_dynabert_forward
    self.base_model_class._ori_forward = self.base_model_class.forward

    def init_func(layer):
        if isinstance(layer, self.base_model_class):
            layer.forward = layer._forward

    for layer in self.children():
        layer.apply(init_func)
    return self


def _replace_auto_model_qat_forward(self):
    self.base_model_class._forward = auto_model_forward
    self.base_model_class._ori_forward = self.base_model_class.forward

    def init_func(layer):
        if isinstance(layer, self.base_model_class):
            layer.forward = layer._forward

    for layer in self.children():
        layer.apply(init_func)
    return self


def _recover_auto_model_forward(self):

    def init_func(layer):
        if isinstance(
                layer, self.base_model_class
                if not isinstance(self, paddle.DataParallel) else
                self._layers.base_model_class):
            layer.forward = layer._ori_forward

    for layer in self._layers.children() if isinstance(
            self, paddle.DataParallel) else self.children():
        layer.apply(init_func)
    return self


def _dynabert_init(self, model, eval_dataloader):
    from paddleslim.nas.ofa.convert_super import Convert, supernet
    from paddleslim.nas.ofa import OFA, DistillConfig, utils

    # Step1: Initialize a dictionary to save the weights from the origin model.
    origin_weights = model.state_dict()

    # Step2: Define teacher model.
    teacher_model = copy.deepcopy(model)

    # Step3: Convert origin model to supernet.
    sp_config = supernet(expand_ratio=[1.0])
    model = Convert(sp_config).convert(model)

    # Use weights saved in the dictionary to initialize supernet.
    utils.set_state_dict(model, origin_weights)
    del origin_weights

    # Step4: Config about distillation.
    mapping_layers = [model.base_model_prefix + '.embeddings']
    for idx in range(model.base_model.config['num_hidden_layers']):
        mapping_layers.append(model.base_model_prefix +
                              '.encoder.layers.{}'.format(idx))

    default_distill_config = {
        'lambda_distill': 0.1,
        'teacher_model': teacher_model,
        'mapping_layers': mapping_layers,
    }
    distill_config = DistillConfig(**default_distill_config)

    # Step5: Config in supernet training.
    ofa_model = OFA(model,
                    distill_config=distill_config,
                    elastic_order=['width'])

    # Step6: Calculate the importance of neurons and head,
    # and then reorder them according to the importance.
    ofa_model.model, ofa_model = _replace_transformer_func(
        ofa_model.model), _replace_transformer_func(ofa_model)
    head_importance, neuron_importance = compute_neuron_head_importance(
        model=ofa_model.model,
        data_loader=eval_dataloader,
        loss_fct=self.criterion,
        num_layers=model.base_model.config['num_hidden_layers'],
        num_heads=model.base_model.config['num_attention_heads'])

    reorder_neuron_head(ofa_model.model, head_importance, neuron_importance)

    if paddle.distributed.get_world_size() > 1:
        ofa_model.model = paddle.DataParallel(ofa_model.model)

    return ofa_model, teacher_model


def check_dynabert_config(net_config, width_mult):
    '''
    Corrects net_config for OFA model if necessary.
    '''
    if 'electra.embeddings_project' in net_config:
        net_config["electra.embeddings_project"]['expand_ratio'] = 1.0
    for key in net_config:
        # Makes sure to expands the size of the last dim to `width_mult` for
        # these Linear weights.
        if 'q_proj' in key or 'k_proj' in key or 'v_proj' in key or 'linear1' in key:
            net_config[key]['expand_ratio'] = width_mult
        # Keeps the size of the last dim of these Linear weights same as
        # before.
        elif 'out_proj' in key or 'linear2' in key:
            net_config[key]['expand_ratio'] = 1.0
    return net_config


def evaluate(self, model, data_loader):
    if self.custom_evaluate is not None:
        return self.custom_evaluate(self, model, data_loader)
    if isinstance(model, paddleslim.nas.ofa.OFA):
        class_name = model.model.__class__.__name__
    else:
        class_name = model.__class__.__name__
    if "SequenceClassification" in class_name:
        return evaluate_seq_cls(self, model, data_loader)
    elif "QuestionAnswering" in class_name:
        return evaluate_qa(self, model, data_loader)
    elif "TokenClassification" in class_name:
        return evaluate_token_cls(self, model, data_loader)
    else:
        raise NotImplementedError(
            "Model to be compressed is an instance of a custom class, " \
            "so function `evaluate(self, model, data_loader)` should be " \
            "implemented, and `model` should support both `paddle.nn.layer` " \
            "and `paddleslim.nas.ofa.OFA` instances, and it should return " \
            "a single float for precision value, such as acc.")


@paddle.no_grad()
def evaluate_qa(self, model, data_loader):
    model.eval()
    all_start_logits = []
    all_end_logits = []
    for batch in data_loader:
        logits = model(input_ids=batch['input_ids'],
                       token_type_ids=batch['token_type_ids'])
        if isinstance(model, paddleslim.nas.ofa.OFA):
            start_logits_tensor, end_logits_tensor = logits[0]
        else:
            start_logits_tensor, end_logits_tensor = logits
        for idx in range(start_logits_tensor.shape[0]):
            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])
    n_best_size = 20
    max_answer_length = 50
    all_predictions, _, _ = compute_prediction(
        self.eval_examples, self.eval_dataset,
        (all_start_logits, all_end_logits), False, n_best_size,
        max_answer_length)
    res = squad_evaluate(examples=[raw_data for raw_data in self.eval_examples],
                         preds=all_predictions,
                         is_whitespace_splited=False)
    logger.info("EM: %f, F1: %f, " % (res['exact'], res['f1']))
    res = res['exact']
    model.train()
    return res


@paddle.no_grad()
def evaluate_seq_cls(self, model, data_loader):
    metric = Accuracy()
    model.eval()
    metric.reset()
    for batch in data_loader:
        labels = batch.pop("labels")
        logits = model(**batch)
        if isinstance(model, paddleslim.nas.ofa.OFA):
            logits = logits[0]
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    logger.info("acc: %s, " % res)
    model.train()
    return res


@paddle.no_grad()
def evaluate_token_cls(self, model, data_loader):
    metric = ChunkEvaluator(label_list=self.train_dataset.label_list)
    model.eval()
    metric.reset()
    for batch in data_loader:
        logits = model(input_ids=batch['input_ids'],
                       token_type_ids=batch['token_type_ids'])
        if isinstance(model, paddleslim.nas.ofa.OFA):
            logits = logits[0]
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            batch['seq_len'], preds, batch['labels'])
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(),
                      num_correct_chunks.numpy())
    res = metric.accumulate()
    logger.info("precision: %f, recall: %f, f1_score: %f" %
                (res[0], res[1], res[2]))
    res = res[2]
    model.train()
    return res


def _dynabert_training(self, ofa_model, model, teacher_model, train_dataloader,
                       eval_dataloader, num_train_epochs):
    from paddleslim.nas.ofa import OFA, DistillConfig, utils
    global_step = 0
    lambda_logit = 1.0
    tic_train = time.time()
    best_acc = [0.0] * len(self.args.width_mult_list)
    acc = 0.0

    logger.info("Teacher's evaluation starts.")
    tic_eval = time.time()
    evaluate(self, teacher_model, eval_dataloader)
    logger.info("eval done total: %s s" % (time.time() - tic_eval))

    logger.info("DynaBERT training starts. This period will cost some time.")
    for epoch in range(num_train_epochs):
        # Step7: Set current epoch and task.
        ofa_model.set_epoch(epoch)
        ofa_model.set_task('width')
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            for width_mult in self.args.width_mult_list:
                # Step8: Broadcast supernet config from width_mult,
                # and use this config in supernet training.
                net_config = utils.dynabert_config(ofa_model, width_mult)
                net_config = check_dynabert_config(net_config, width_mult)
                ofa_model.set_net_config(net_config)
                if "token_type_ids" in batch:
                    logits, teacher_logits = ofa_model(
                        input_ids=batch['input_ids'],
                        token_type_ids=batch['token_type_ids'],
                        attention_mask=[None, None])
                else:
                    logits, teacher_logits = ofa_model(
                        batch['input_ids'], attention_mask=[None, None])
                rep_loss = ofa_model.calc_distill_loss()
                if isinstance(logits, tuple):
                    logit_loss = 0
                    for i in range(len(logits)):
                        logit_loss += soft_cross_entropy(
                            logits[i], teacher_logits[i].detach())
                    logit_loss /= len(logits)
                else:
                    logit_loss = soft_cross_entropy(logits,
                                                    teacher_logits.detach())
                loss = rep_loss + lambda_logit * logit_loss
                loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.clear_grad()

            if global_step % self.args.logging_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss,
                           self.args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % self.args.save_steps == 0:
                for idx, width_mult in enumerate(self.args.width_mult_list):
                    net_config = utils.dynabert_config(ofa_model, width_mult)
                    net_config = check_dynabert_config(net_config, width_mult)
                    ofa_model.set_net_config(net_config)
                    tic_eval = time.time()
                    logger.info("width_mult %s:" % round(width_mult, 2))
                    acc = evaluate(self, ofa_model, eval_dataloader)
                    if acc > best_acc[idx]:
                        best_acc[idx] = acc
                        if paddle.distributed.get_rank() == 0:
                            output_dir_width = os.path.join(
                                self.args.output_dir,
                                "width_mult_" + str(round(width_mult, 2)))
                            if not os.path.exists(output_dir_width):
                                os.makedirs(output_dir_width)
                            # need better way to get inner model of DataParallel
                            model_to_save = model._layers if isinstance(
                                model, paddle.DataParallel) else model
                            model_to_save.save_pretrained(output_dir_width)
                    logger.info("eval done total: %s s" %
                                (time.time() - tic_eval))
            if global_step > self.args.num_training_steps:
                if best_acc[idx] == 0.0:
                    output_dir_width = os.path.join(
                        self.args.output_dir,
                        "width_mult_" + str(round(width_mult, 2)))
                    if not os.path.exists(output_dir_width):
                        os.makedirs(output_dir_width)
                    # need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir_width)
                logger.info("Best result of width_mult %.2f: %.4f" %
                            (width_mult, best_acc[idx]))
                return ofa_model

    for idx, width_mult in enumerate(self.args.width_mult_list):
        logger.info("Best result of width_mult %.2f: %.4f" %
                    (width_mult, best_acc[idx]))
    return ofa_model


def _dynabert_export(self, ofa_model):
    from paddleslim.nas.ofa import OFA, DistillConfig, utils
    ofa_model._add_teacher = False
    ofa_model, ofa_model.model = _recover_transformer_func(
        ofa_model), _recover_transformer_func(ofa_model.model)
    if isinstance(ofa_model.model, paddle.DataParallel):
        ori_num_heads = ofa_model.model._layers.base_model.encoder.layers[
            0].self_attn.num_heads
    else:
        ori_num_heads = ofa_model.model.base_model.encoder.layers[
            0].self_attn.num_heads
    for width_mult in self.args.width_mult_list:
        model_dir = os.path.join(self.args.output_dir,
                                 "width_mult_" + str(round(width_mult, 2)))
        state_dict = paddle.load(os.path.join(model_dir,
                                              "model_state.pdparams"))
        origin_model = self.model.__class__.from_pretrained(model_dir)
        ofa_model.model.set_state_dict(state_dict)
        best_config = utils.dynabert_config(ofa_model, width_mult)
        best_config = check_dynabert_config(best_config, width_mult)
        input_spec = generate_input_spec(self.model, self.train_dataset)
        origin_model_new = ofa_model.export(
            best_config,
            input_shapes=[[1, 1]] * len(input_spec),
            input_dtypes=['int64'] * len(input_spec),
            origin_model=origin_model)
        for name, sublayer in origin_model_new.named_sublayers():
            if isinstance(sublayer, paddle.nn.MultiHeadAttention):
                sublayer.num_heads = int(width_mult * sublayer.num_heads)

        pruned_infer_model_dir = os.path.join(model_dir, "pruned_model")
        net = paddle.jit.to_static(origin_model_new, input_spec=input_spec)
        paddle.jit.save(net, pruned_infer_model_dir)
        # Recover num_heads of ofa_model.model
        if isinstance(ofa_model.model, paddle.DataParallel):
            for layer in ofa_model.model._layers.base_model.encoder.layers:
                layer.self_attn.num_heads = ori_num_heads
        else:
            for layer in ofa_model.model.base_model.encoder.layers:
                layer.self_attn.num_heads = ori_num_heads
    logger.info("Pruned models have been exported.")


def _post_training_quantization_grid_search(self, model_dir):
    eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
    args = self.args
    if args.batch_num_list is None:
        args.batch_num_list = [1]
    if args.batch_size_list is None:
        args.batch_size_list = [4, 8, 16]
    if args.algo_list is None:
        args.algo_list = ['mse', 'KL']

    paddle.enable_static()
    place = paddle.set_device(args.device)
    exe = paddle.static.Executor(place)

    args.output_filename_prefix = "int8"

    def _post_training_quantization(algo, batch_size, batch_nums):

        def _batch_generator_func():
            param_name_list = []
            for key in self.eval_dataset[0]:
                if key in ('input_ids', 'token_type_ids'):
                    param_name_list.append(key)
            batch_data = [[] for i in range(len(param_name_list))]
            for data in self.eval_dataset:
                for i in range(len(param_name_list)):
                    batch_data[i].append(data[param_name_list[i]])
                if len(batch_data[0]) == batch_size:
                    for i in range(len(param_name_list)):
                        batch_data[i] = Pad(axis=0, pad_val=0)(batch_data[i])
                    yield batch_data
                    batch_data = [[] for i in range(len(param_name_list))]

        post_training_quantization = PostTrainingQuantization(
            executor=exe,
            batch_generator=_batch_generator_func,
            model_dir=model_dir,
            model_filename=args.input_filename_prefix + ".pdmodel",
            params_filename=args.input_filename_prefix + ".pdiparams",
            batch_size=batch_size,
            batch_nums=batch_nums,
            scope=None,
            algo=algo,
            hist_percent=0.9999,
            round_type=args.round_type,
            bias_correction=args.bias_correction,
            quantizable_op_type=['matmul', 'matmul_v2'],
            is_full_quantize=False,
            weight_bits=8,
            activation_bits=8,
            activation_quantize_type='range_abs_max'
            if args.activation_quantize_type is None else
            args.activation_quantize_type,
            weight_quantize_type=args.weight_quantize_type,
            onnx_format=False,
            optimize_model=False)
        post_training_quantization.quantize()
        post_training_quantization.save_quantized_model(
            save_model_path=os.path.join(
                model_dir, algo +
                "_".join([str(batch_size), str(batch_nums)])),
            model_filename=args.output_filename_prefix + ".pdmodel",
            params_filename=args.output_filename_prefix + ".pdiparams")

    logger.info("Post training quantization starts.")
    for algo in args.algo_list:
        for batch_size in args.batch_size_list:
            for batch_nums in args.batch_num_list:
                _post_training_quantization(algo, batch_size, batch_nums)

    paddle.disable_static()
    logger.info(
        "Post training quantization ends and quantized models are saved.")


def _quant_aware_training_dynamic(self, input_dir):
    args = self.args
    args.output_filename_prefix = "int8"
    quant_config = {
        'activation_preprocess_type':
        args.activation_preprocess_type,  # PACT or None
        'weight_preprocess_type':
        args.weight_preprocess_type,  # PACT or None
        'weight_quantize_type':
        args.weight_quantize_type,
        'activation_quantize_type':
        'moving_average_abs_max' if args.activation_quantize_type is None else
        args.activation_quantize_type,
        'weight_bits':
        8,
        'activation_bits':
        8,
        'dtype':
        'int8',
        # window size for 'range_abs_max' quantization. defaulf is 10000
        'window_size':
        10000,
        'quantizable_layer_type': ['Linear', 'Conv2D'],
        'moving_rate':
        args.moving_rate,
    }

    from paddleslim import QAT
    self.model = _replace_auto_model_qat_forward(self.model)

    quanter = QAT(config=quant_config)

    quanter.quantize(self.model)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_param_path = os.path.join(args.output_dir, "best_quant.pdparams")

    train_dataloader = self.get_train_dataloader()
    eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

    # TODO: args.gradient_accumulation_steps
    if args.max_steps > 0:
        args.num_training_steps = args.max_steps
        args.num_train_epochs = math.ceil(num_training_steps /
                                          len(train_dataloader))
    else:
        args.num_training_steps = len(train_dataloader) * args.num_train_epochs
        args.num_train_epochs = math.ceil(args.num_train_epochs)

    self.create_optimizer_and_scheduler(
        num_training_steps=args.num_training_steps)

    global_step = 0
    tic_train = time.time()
    best_acc, acc = 0.0, 0.0

    logger.info("FP32 model's evaluation starts.")

    tic_eval = time.time()
    acc = evaluate(self, self.model, eval_dataloader)
    logger.info("eval done total: %s s" % (time.time() - tic_eval))
    logger.info("Quant aware training starts.")
    # Train self.model
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            labels = None
            if "labels" in batch:
                labels = batch.pop("labels")
            elif "start_positions" in batch and "end_positions" in batch:
                labels = (batch.pop("start_positions"),
                          batch.pop("end_positions"))
            model_para_keys = inspect.signature(
                self.model.forward).parameters.keys()
            inputs = {}
            for key in batch:
                if key in model_para_keys:
                    inputs[key] = batch[key]
            # import pdb; pdb.set_trace()
            logits = self.model(**inputs)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.clear_grad()
            if global_step % self.args.logging_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss, args.logging_steps /
                           (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0:
                tic_eval = time.time()
                acc = evaluate(self, self.model, eval_dataloader)
                if acc > best_acc:
                    best_acc = acc
                    if paddle.distributed.get_rank() == 0:
                        # need better way to get inner model of DataParallel
                        model_to_save = self.model._layers if isinstance(
                            self.model, paddle.DataParallel) else self.model
                        paddle.save(model_to_save.state_dict(),
                                    output_param_path)
                logger.info("eval done total: %s s" % (time.time() - tic_eval))
    logger.info("Best result: %.4f" % best_acc)
    self.model.set_state_dict(paddle.load(output_param_path))

    input_spec = generate_input_spec(self.model, self.train_dataset)

    quanter.save_quantized_model(self.model,
                                 os.path.join(input_dir,
                                              args.output_filename_prefix),
                                 input_spec=input_spec,
                                 onnx_format=True)
    self.model = _recover_auto_model_forward(self.model)
    logger.info("Quant aware training ends and quantized models are saved.")


def auto_model_dynabert_forward(self,
                                input_ids,
                                token_type_ids=None,
                                position_ids=None,
                                attention_mask=[None, None],
                                task_type_ids=None,
                                past_key_values=None,
                                inputs_embeds=None,
                                use_cache=None,
                                output_hidden_states=False,
                                output_attentions=False,
                                return_dict=False):
    kwargs = locals()
    wtype = self.encoder.layers[0].norm1.fn.weight.dtype if hasattr(
        self.encoder.layers[0].norm1,
        'fn') else self.encoder.layers[0].norm1.weight.dtype
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time."
        )
    elif input_ids is not None:
        input_shape = paddle.shape(input_ids)
    elif inputs_embeds is not None:
        input_shape = paddle.shape(inputs_embeds)[:-1]
    else:
        raise ValueError(
            "You have to specify either input_ids or inputs_embeds")

    past_key_values_length = None
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if attention_mask is None:
        # input_ids[0][0] is equals to 0 while exporting.
        if input_ids[0][0] != 0:
            attention_mask = [None, None]
            attention_mask[0] = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(wtype) * -1e4,
                axis=[1, 2])
        else:
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = paddle.zeros(
                    [batch_size, 1, 1, past_key_values_length],
                    dtype=attention_mask.dtype)
                attention_mask = paddle.concat([past_mask, attention_mask],
                                               axis=-1)
    elif isinstance(attention_mask, paddle.Tensor) and attention_mask.ndim == 2:
        attention_mask = paddle.unsqueeze(attention_mask,
                                          axis=[1, 2]).astype(wtype)
        attention_mask = (1.0 - attention_mask) * -1e4
    elif attention_mask[0] is None:
        attention_mask[0] = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(wtype) * -1e4, axis=[1, 2])

    embedding_kwargs_keys = inspect.signature(
        self.embeddings.forward).parameters.keys()
    embedding_kwargs = {}
    for key in embedding_kwargs_keys:
        if key in kwargs.keys():
            embedding_kwargs[key] = kwargs[key]
    embedding_kwargs["input_ids"] = input_ids

    embedding_output = self.embeddings(**embedding_kwargs)
    if hasattr(self, "embeddings_project"):
        embedding_output = self.embeddings_project(embedding_output)

    self.encoder._use_cache = use_cache  # To be consistent with HF

    encoder_kwargs_keys = inspect.signature(
        self.encoder.forward).parameters.keys()
    encoder_kwargs = {}
    for key in encoder_kwargs_keys:
        if key == "cache":
            encoder_kwargs[key] = past_key_values
        elif key == "src_mask":
            encoder_kwargs[key] = attention_mask
        elif key in kwargs:
            encoder_kwargs[key] = kwargs[key]

    encoder_outputs = self.encoder(embedding_output, **encoder_kwargs)
    if isinstance(encoder_outputs, type(embedding_output)):
        sequence_output = encoder_outputs
        if hasattr(self, 'pooler'):
            pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = sequence_output[:, 0]
        return (sequence_output, pooled_output)
    else:
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions)


def auto_model_forward(self,
                       input_ids,
                       token_type_ids=None,
                       position_ids=None,
                       attention_mask=None,
                       task_type_ids=None,
                       past_key_values=None,
                       inputs_embeds=None,
                       use_cache=None,
                       output_hidden_states=False,
                       output_attentions=False,
                       return_dict=False):
    past_key_values_length = None
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if attention_mask is None:
        attention_mask = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(paddle.float32),
            axis=[1, 2])
        if past_key_values is not None:
            batch_size = past_key_values[0][0].shape[0]
            past_mask = paddle.zeros([batch_size, 1, 1, past_key_values_length],
                                     dtype=attention_mask.dtype)
            attention_mask = paddle.concat([past_mask, attention_mask], axis=-1)
    # For 2D attention_mask from tokenizer
    elif attention_mask.ndim == 2:
        attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(
            paddle.get_default_dtype())
        attention_mask = (1.0 - attention_mask) * -1e4
    return self._ori_forward(input_ids,
                             token_type_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             task_type_ids=task_type_ids,
                             past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds,
                             use_cache=use_cache,
                             output_hidden_states=output_hidden_states,
                             output_attentions=output_attentions,
                             return_dict=return_dict)


def soft_cross_entropy(inp, target):
    inp_likelihood = F.log_softmax(inp, axis=-1)
    target_prob = F.softmax(target, axis=-1)
    return -1. * paddle.mean(paddle.sum(inp_likelihood * target_prob, axis=-1))


Trainer.compress = compress
Trainer.quant = quant

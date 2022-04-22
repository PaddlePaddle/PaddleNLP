# Copyright 2020-present the HuggingFace Inc. team.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization

nn.MultiHeadAttention._ori_forward = paddle.nn.MultiHeadAttention.forward
nn.MultiHeadAttention._ori_prepare_qkv = nn.MultiHeadAttention._prepare_qkv

from paddlenlp.trainer import Trainer
from paddlenlp.utils.log import logger
from paddlenlp.data import Pad
from paddlenlp.transformers import AutoModelForSequenceClassification, ErnieModel


def try_import_paddleslim():
    try:
        import paddleslim
    except ImportError:
        raise ImportError(
            'Cannot import paddleslim, please install paddleslim.')


class DynabertConfig:
    def __init__(self,
                 width_mult_list=[3 / 4],
                 output_filename_prefix="float32"):
        self.compress_type = "dynabert"
        self.width_mult_list = width_mult_list
        self.output_filename_prefix = output_filename_prefix


class PTQConfig:
    def __init__(self,
                 algo="hist",
                 batch_size=4,
                 input_dir=None,
                 output_dir=None,
                 input_filename_prefix="float32",
                 output_filename_prefix="int8"):
        self.compress_type = "ptq"
        self.algo = algo
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_filename_prefix = input_filename_prefix
        self.output_filename_prefix = output_filename_prefix


class CompressConfig:
    def __init__(self,
                 prune_config=DynabertConfig(),
                 quantization_config=PTQConfig()):
        assert isinstance(prune_config, (DynabertConfig)), \
           "`prune_config` should be an instance of `DynabertConfig`."
        assert isinstance(quantization_config, (PTQConfig)), \
           "`quantization_config` shoule be an instance of `PTQConfig`."
        self.prune_config = prune_config
        self.quantization_config = quantization_config


def compress(self,
             output_dir,
             pruning=True,
             quantization=True,
             compress_config=CompressConfig()):
    """
    Supports pruning and quantization. If both are needed, pruning would be
    performed before quantizaton.
    """
    if pruning:
        try_import_paddleslim()
        self.prune(output_dir, compress_config.prune_config)
        if quantization:
            for width_mult in compress_config.prune_config.width_mult_list:
                output_dir_width = os.path.join(output_dir, str(width_mult))
                self.quant(output_dir_width, output_dir,
                           compress_config.quantization_config)
    elif self.quantization:
        input_dir = output_dir
        if compress_config.quantization_config.input_dir is None:
            pass  # 先动转静保存
            input_dir = compress_config.quantization_config.input_dir
        self.quant(input_dir, output_dir, compress_config.quantization_config)


def prune(self, output_dir, prune_config=DynabertConfig()):
    """
    Supports DynaBERT strategy.
    """
    if prune_config.compress_type == "dynabert":
        _dynabert(self, self.model, output_dir, prune_config)


def quant(self, input_dir, output_dir, quantization_config=PTQConfig()):
    """
    Supports Post-Training Quantization now.
    """
    assert quantization_config.compress_type == "ptq", \
        "Only Post Training Quantization is supported."
    eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
    nn.MultiHeadAttention._prepare_qkv = nn.MultiHeadAttention._ori_prepare_qkv
    _post_training_quantization(eval_dataloader, self.eval_dataset, input_dir,
                                output_dir, quantization_config)


def _dynabert(self, model, output_dir, dynabert_config):
    model.base_model_class._ori_forward = model.base_model_class.forward
    model.base_model_class.forward = auto_model_forward

    criterion = paddle.nn.loss.CrossEntropyLoss()
    # Each batch is a dict.
    train_dataloader = self.get_train_dataloader()
    eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

    ofa_model, teacher_model = _dynabert_init(model, eval_dataloader, criterion,
                                              dynabert_config.width_mult_list)

    args = self.args
    # TODO: args.gradient_accumulation_steps
    if args.max_steps > 0:
        args.num_training_steps = args.max_steps
        args.num_train_epochs = math.ceil(num_training_steps /
                                          len(train_dataloader))
    else:
        args.num_training_steps = len(train_dataloader) * args.num_train_epochs
        args.num_train_epochs = args.num_train_epochs
    self.create_optimizer_and_scheduler(
        num_training_steps=args.num_training_steps)

    metric = paddle.metric.Accuracy()

    ofa_model = _dynabert_training(self, ofa_model, model, teacher_model,
                                   train_dataloader, eval_dataloader,
                                   dynabert_config.width_mult_list, criterion,
                                   metric, args.num_train_epochs, output_dir)

    # Each width_mult best model would be exported.
    _dynabert_export(ofa_model, dynabert_config, output_dir)

    model.base_model_class.forward = model.base_model_class._ori_forward
    logger.info("DynaBERT training finished.")


def _recover_transormer_func():
    nn.TransformerEncoder.forward = paddle.nn.TransformerEncoder._ori_forward
    nn.TransformerEncoderLayer.forward = paddle.nn.TransformerEncoderLayer._ori_forward
    nn.MultiHeadAttention.forward = paddle.nn.MultiHeadAttention._ori_forward
    # nn.MultiHeadAttention._prepare_qkv = nn.MultiHeadAttention._ori_prepare_qkv


def _dynabert_init(model, eval_dataloader, criterion, width_mult_list):
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
    # NOTE: Importing `nlp_utils` would rewrite `forward` function of
    # TransformerEncoder, TransformerEncoderLayer, MultiHeadAttention and
    # `_prepare_qkv` function of MultiHeadAttention.
    from paddleslim.nas.ofa.utils import nlp_utils

    head_importance, neuron_importance = nlp_utils.compute_neuron_head_importance(
        task_name="",
        model=ofa_model.model,
        data_loader=eval_dataloader,
        loss_fct=criterion,
        num_layers=model.base_model.config['num_hidden_layers'],
        num_heads=model.base_model.config['num_attention_heads'])
    reorder_neuron_head(ofa_model.model, head_importance, neuron_importance)

    if paddle.distributed.get_world_size() > 1:
        ofa_model.model = paddle.DataParallel(ofa_model.model)

    return ofa_model, teacher_model


def _dynabert_training(self, ofa_model, model, teacher_model, train_dataloader,
                       eval_dataloader, width_mult_list, criterion, metric,
                       num_train_epochs, output_dir):
    @paddle.no_grad()
    def evaluate(model, criterion, metric, data_loader, width_mult=1.0):
        model.eval()
        metric.reset()
        for batch in data_loader:
            input_ids, segment_ids, labels = batch['input_ids'], batch[
                'token_type_ids'], batch['labels']
            logits = model(input_ids, segment_ids, attention_mask=[None, None])
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, labels)
            correct = metric.compute(logits, labels)
            metric.update(correct)
        res = metric.accumulate()
        # Teacher model's evaluation
        if width_mult == 100:
            logger.info("teacher model, eval loss: %f, acc: %s, " %
                        (loss.numpy(), res))
        else:
            logger.info("width_mult: %s, eval loss: %f, acc: %s, " %
                        (str(width_mult), loss.numpy(), res))
        model.train()
        return res

    from paddleslim.nas.ofa import OFA, DistillConfig, utils
    global_step = 0
    lambda_logit = 1.0
    tic_train = time.time()

    best_acc = 0.0
    logger.info("DynaBERT training starts. This period will cost some time.")
    for epoch in range(num_train_epochs):
        # Step7: Set current epoch and task.
        ofa_model.set_epoch(epoch)
        ofa_model.set_task('width')

        for step, batch in enumerate(train_dataloader):
            global_step += 1
            input_ids, segment_ids, labels = batch['input_ids'], batch[
                'token_type_ids'], batch['labels']

            for width_mult in width_mult_list:
                # Step8: Broadcast supernet config from width_mult,
                # and use this config in supernet training.
                net_config = utils.dynabert_config(ofa_model, width_mult)
                ofa_model.set_net_config(net_config)
                logits, teacher_logits = ofa_model(
                    input_ids, segment_ids, attention_mask=[None, None])
                rep_loss = ofa_model.calc_distill_loss()
                logit_loss = soft_cross_entropy(logits, teacher_logits.detach())
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
                tic_eval = time.time()
                evaluate(
                    teacher_model,
                    criterion,
                    metric,
                    eval_dataloader,
                    width_mult=100)
                logger.info("eval done total : %s s" % (time.time() - tic_eval))
                for idx, width_mult in enumerate(width_mult_list):
                    net_config = utils.dynabert_config(ofa_model, width_mult)
                    ofa_model.set_net_config(net_config)
                    tic_eval = time.time()
                    acc = evaluate(ofa_model, criterion, metric,
                                   eval_dataloader, width_mult)
                    if acc > best_acc:
                        best_acc = acc
                    if paddle.distributed.get_rank() == 0:
                        output_dir_width = os.path.join(output_dir,
                                                        str(width_mult))
                        if not os.path.exists(output_dir_width):
                            os.makedirs(output_dir_width)
                        # need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir_width)
                    logger.info("eval done total : %s s" %
                                (time.time() - tic_eval))
            if global_step >= self.args.num_training_steps:
                logger.info("Best acc: %.2f" % (best_acc))
                return ofa_model
    logger.info("Best acc: %.2f" % (best_acc))
    return ofa_model


def _dynabert_export(ofa_model, dynabert_config, output_dir):
    from paddleslim.nas.ofa import OFA, DistillConfig, utils
    ofa_model.model.base_model_class.forward = auto_model_forward
    ofa_model._add_teacher = False
    _recover_transormer_func()

    for width_mult in dynabert_config.width_mult_list:
        model_dir = os.path.join(output_dir, str(width_mult))
        state_dict = paddle.load(
            os.path.join(model_dir, "model_state.pdparams"))
        # TODO: Supports more tasks.
        origin_model = AutoModelForSequenceClassification.from_pretrained(
            model_dir)
        ofa_model.model.set_state_dict(state_dict)
        best_config = utils.dynabert_config(ofa_model, width_mult)
        origin_model_new = ofa_model.export(
            best_config,
            input_shapes=[[1, 1], [1, 1]],
            input_dtypes=['int64', 'int64'],
            origin_model=origin_model)

        for name, sublayer in origin_model_new.named_sublayers():
            if isinstance(sublayer, paddle.nn.MultiHeadAttention):
                sublayer.num_heads = int(width_mult * sublayer.num_heads)

        input_shape = [
            paddle.static.InputSpec(
                shape=[None, None], dtype='int64'), paddle.static.InputSpec(
                    shape=[None, None], dtype='int64')
        ]
        pruned_infer_model_dir = os.path.join(
            model_dir, dynabert_config.output_filename_prefix)
        net = paddle.jit.to_static(origin_model_new, input_spec=input_shape)
        paddle.jit.save(net, pruned_infer_model_dir)


def _post_training_quantization(eval_dataloader, eval_dataset, input_dir,
                                output_dir, quantization_config):
    paddle.enable_static()
    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)

    def batch_generator_func():
        batch_data = [[], []]
        for data in eval_dataset:
            batch_data[0].append(data['input_ids'])
            batch_data[1].append(data['token_type_ids'])
            if len(batch_data[0]) == quantization_config.batch_size:
                input_ids = Pad(axis=0, pad_val=0)(batch_data[0])
                segment_ids = Pad(axis=0, pad_val=0)(batch_data[1])
                yield [input_ids, segment_ids]
                batch_data = [[], []]

    logger.info("Post training quantization starts.")
    post_training_quantization = PostTrainingQuantization(
        executor=exe,
        batch_generator=batch_generator_func,
        # data_loader=data_loader,
        model_dir=input_dir,
        model_filename=quantization_config.input_filename_prefix + ".pdmodel",
        params_filename=quantization_config.input_filename_prefix +
        ".pdiparams",
        batch_size=quantization_config.batch_size,
        batch_nums=1,
        scope=None,
        algo=quantization_config.algo,
        # round_type='round',
        hist_percent=0.9999,
        bias_correction=False,
        quantizable_op_type=['matmul', 'matmul_v2'],
        is_full_quantize=False,
        weight_bits=8,
        activation_bits=8,
        activation_quantize_type='range_abs_max',
        weight_quantize_type='channel_wise_abs_max',
        # onnx_format=True,
        optimize_model=False)
    post_training_quantization.quantize()

    post_training_quantization.save_quantized_model(
        save_model_path=os.path.join(
            output_dir,
            quantization_config.algo + str(quantization_config.batch_size)),
        model_filename=quantization_config.output_filename_prefix + ".pdmodel",
        params_filename=quantization_config.output_filename_prefix +
        ".pdiparams")

    logger.info("Post training quantization ends.")


def static_eval(self):
    """Evaluates models after quantization"""
    # paddle.enable_static()
    pass
    # paddle.disable_static()


def auto_model_forward(self,
                       input_ids,
                       token_type_ids=None,
                       position_ids=None,
                       attention_mask=[None, None]):
    wtype = self.pooler.dense.fn.weight.dtype if hasattr(
        self.pooler.dense, 'fn') else self.pooler.dense.weight.dtype
    if attention_mask is None:
        attention_mask = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(wtype) * -1e9, axis=[1, 2])
    if attention_mask[0] is None:
        attention_mask[0] = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(wtype) * -1e9, axis=[1, 2])
    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids)
    encoder_outputs = self.encoder(embedding_output, attention_mask)
    sequence_output = encoder_outputs
    pooled_output = self.pooler(sequence_output)
    return sequence_output, pooled_output


def reorder_neuron_head(model, head_importance, neuron_importance):
    """
    Reorders weights according head importance and neuron importance
    """
    from paddleslim.nas.ofa.utils import nlp_utils
    # Reorders heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # Reorders heads
        idx = paddle.argsort(head_importance[layer], descending=True)
        nlp_utils.reorder_head(model.base_model.encoder.layers[layer].self_attn,
                               idx)
        # Reorders neurons
        idx = paddle.argsort(
            paddle.to_tensor(current_importance), descending=True)
        nlp_utils.reorder_neuron(
            model.base_model.encoder.layers[layer].linear1.fn, idx, dim=1)

        nlp_utils.reorder_neuron(
            model.base_model.encoder.layers[layer].linear2.fn, idx, dim=0)


def soft_cross_entropy(inp, target):
    inp_likelihood = F.log_softmax(inp, axis=-1)
    target_prob = F.softmax(target, axis=-1)
    return -1. * paddle.mean(paddle.sum(inp_likelihood * target_prob, axis=-1))


Trainer.compress = compress
Trainer.prune = prune
Trainer.quant = quant

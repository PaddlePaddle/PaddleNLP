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
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# Paddleslim will modify MultiHeadAttention.forward and MultiHeadAttention._prepare_qkv
# Original forward and _prepare_qkv should be saved before import paddleslim
nn.MultiHeadAttention._ori_forward = paddle.nn.MultiHeadAttention.forward
nn.MultiHeadAttention._ori_prepare_qkv = nn.MultiHeadAttention._prepare_qkv

from paddlenlp.trainer import Trainer
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoModelForSequenceClassification
from paddlenlp.transformers import export_model

from metric import MetricReport


def try_import_paddleslim():
    """
    Try to import paddleslim
    """
    try:
        import paddleslim
    except ImportError:
        raise ImportError(
            'Cannot import paddleslim, please install paddleslim.')


class DynabertConfig(object):
    """
    Define DynaBERT configuration class
    """

    def __init__(self, width_mult=2 / 3, output_filename_prefix="float32"):
        """
        Pruning class config of DynaBERT stratedy.
        Args:
            width_mult (float):
                Width mult for DynaBERT.
                Defaults to `2/3`.
            output_filename_prefix (str):
                Prefix of pruned model's filename. 
                Defaults to `float32`.
        """
        self.compress_type = "dynabert"
        self.width_mult = width_mult
        self.output_filename_prefix = output_filename_prefix


def prune(self, output_dir, prune_config):
    """
    Supports pruning (DynaBERT) now.
    Args:
        output_dir (str):
            Directory name of Pruning.
        prune_config (`DynabertConfig`):
            Prune config instance to pass parameters for pruning.
            Defaults to `DynabertConfig()`.
    """
    assert isinstance(prune_config, (DynabertConfig)), \
        "`prune_config` should be an instance of `DynabertConfig`."
    try_import_paddleslim()
    logger.info("Pruning starts.")
    if prune_config.compress_type == "dynabert":
        _dynabert(self, self.model, output_dir, prune_config)


def _dynabert(self, model, output_dir, dynabert_config):
    model.base_model_class._ori_forward = model.base_model_class.forward
    model.base_model_class.forward = auto_model_forward

    # Each batch is a dict.
    train_dataloader = self.get_train_dataloader()
    eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

    ofa_model, teacher_model = _dynabert_init(model, eval_dataloader,
                                              self.criterion,
                                              dynabert_config.width_mult)

    args = self.args
    args.num_training_steps = len(train_dataloader) * args.num_train_epochs
    args.num_train_epochs = args.num_train_epochs

    self.create_optimizer_and_scheduler(
        num_training_steps=args.num_training_steps)

    ofa_model = _dynabert_training(self, ofa_model, model, teacher_model,
                                   train_dataloader, eval_dataloader,
                                   dynabert_config.width_mult, self.criterion,
                                   args.num_train_epochs, output_dir)

    # Each width_mult best model would be exported.
    _dynabert_export(ofa_model, dynabert_config, output_dir)

    model.base_model_class.forward = model.base_model_class._ori_forward
    logger.info("DynaBERT training finished.")


def _recover_transormer_func():
    nn.TransformerEncoder.forward = paddle.nn.TransformerEncoder._ori_forward
    nn.TransformerEncoderLayer.forward = paddle.nn.TransformerEncoderLayer._ori_forward
    nn.MultiHeadAttention.forward = paddle.nn.MultiHeadAttention._ori_forward
    # nn.MultiHeadAttention._prepare_qkv = nn.MultiHeadAttention._ori_prepare_qkv


def _dynabert_init(model, eval_dataloader, criterion, width_mult):
    from paddleslim.nas.ofa.convert_super import Convert, supernet
    from paddleslim.nas.ofa import OFA, DistillConfig, utils

    origin_weights = model.state_dict()

    teacher_model = copy.deepcopy(model)

    sp_config = supernet(expand_ratio=[1.0])
    model = Convert(sp_config).convert(model)

    # Use weights saved in the dictionary to initialize supernet.
    utils.set_state_dict(model, origin_weights)
    del origin_weights

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

    ofa_model = OFA(model,
                    distill_config=distill_config,
                    elastic_order=['width'])

    # NOTE: Importing `nlp_utils` would rewrite `forward` function of
    # TransformerEncoder, TransformerEncoderLayer, MultiHeadAttention and
    # `_prepare_qkv` function of MultiHeadAttention.
    from paddleslim.nas.ofa.utils import nlp_utils

    head_importance, neuron_importance = compute_neuron_head_importance(
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
                       eval_dataloader, width_mult, criterion, num_train_epochs,
                       output_dir):
    metric = MetricReport()

    @paddle.no_grad()
    def evaluate(model, criterion, data_loader, width_mult=1.0):
        """
        Given a dataset, it evals model and computes the metric.
        Args:
            model(obj:`paddle.nn.Layer`): A model to classify texts.
            criterion(obj:`paddle.nn.Layer`): It can compute the loss.
            data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
            width_mult(obj:`int/float): width.
        """
        model.eval()
        metric.reset()
        losses = []
        for batch in data_loader:

            input_ids, segment_ids, labels = batch['input_ids'], batch[
                'token_type_ids'], batch['labels']
            logits = model(input_ids, segment_ids, attention_mask=[None, None])
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.sigmoid(logits)
            loss = criterion(logits, labels)
            losses.append(loss.numpy())
            metric.update(probs, labels)

        micro_f1_score, macro_f1_score = metric.accumulate()

        if width_mult == 100:
            logger.info(
                "teacher model, eval loss: %.5f, micro f1 score: %.5f, macro f1 score: %.5f"
                % (np.mean(losses), micro_f1_score, macro_f1_score))
        else:
            logger.info(
                "width_mult: %s, eval loss: %.5f, micro f1 score: %.5f, macro f1 score: %.5f"
                % (str(width_mult), np.mean(losses), micro_f1_score,
                   macro_f1_score))
        model.train()
        return macro_f1_score

    from paddleslim.nas.ofa import OFA, DistillConfig, utils
    global_step = 0
    lambda_logit = 1.0
    tic_train = time.time()
    best_macro_f1_score = 0.0
    logger.info("DynaBERT training starts. This period will cost some time.")
    for epoch in range(num_train_epochs):
        ofa_model.set_epoch(epoch)
        ofa_model.set_task('width')

        for step, batch in enumerate(train_dataloader):
            global_step += 1
            input_ids, token_type_ids, labels = batch['input_ids'], batch[
                'token_type_ids'], batch['labels']

            net_config = utils.dynabert_config(ofa_model, width_mult)

            ofa_model.set_net_config(net_config)
            logits, teacher_logits = ofa_model(input_ids,
                                               token_type_ids,
                                               attention_mask=[None, None])
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

                evaluate(teacher_model,
                         criterion,
                         eval_dataloader,
                         width_mult=100)
                logger.info("eval done total : %s s" % (time.time() - tic_eval))

                net_config = utils.dynabert_config(ofa_model, width_mult)
                ofa_model.set_net_config(net_config)
                tic_eval = time.time()
                macro_f1_score = evaluate(ofa_model, criterion, eval_dataloader,
                                          width_mult)
                if macro_f1_score > best_macro_f1_score:
                    best_macro_f1_score = macro_f1_score
                    if paddle.distributed.get_rank() == 0:
                        output_dir_width = os.path.join(output_dir,
                                                        str(width_mult))
                        if not os.path.exists(output_dir_width):
                            os.makedirs(output_dir_width)
                        # need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir_width)
                logger.info("eval done total : %s s" % (time.time() - tic_eval))

            if global_step > self.args.num_training_steps:
                if best_macro_f1_score == 0.0:
                    output_dir_width = os.path.join(output_dir, str(width_mult))
                    if not os.path.exists(output_dir_width):
                        os.makedirs(output_dir_width)
                    # need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir_width)
                logger.info("Best macro_f1_score: %.4f" % (best_macro_f1_score))
                return ofa_model

    logger.info("width_mult: %s, Best macro f1 score: %.4f" %
                (str(width_mult), best_macro_f1_score))
    return ofa_model


def _dynabert_export(ofa_model, dynabert_config, output_dir):
    from paddleslim.nas.ofa import OFA, DistillConfig, utils
    ofa_model.model.base_model_class.forward = auto_model_forward
    ofa_model._add_teacher = False
    _recover_transormer_func()

    width_mult = dynabert_config.width_mult
    model_dir = os.path.join(output_dir, str(width_mult))
    state_dict = paddle.load(os.path.join(model_dir, "model_state.pdparams"))
    origin_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    ofa_model.model.set_state_dict(state_dict)
    best_config = utils.dynabert_config(ofa_model, width_mult)

    origin_model_new = ofa_model.export(best_config,
                                        input_shapes=[[1, 1], [1, 1]],
                                        input_dtypes=['int64', 'int64'],
                                        origin_model=origin_model)

    for name, sublayer in origin_model_new.named_sublayers():
        if isinstance(sublayer, paddle.nn.MultiHeadAttention):
            sublayer.num_heads = int(width_mult * sublayer.num_heads)

    input_shape = [
        paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        paddle.static.InputSpec(shape=[None, None], dtype='int64')
    ]
    pruned_infer_model_dir = os.path.join(
        model_dir, dynabert_config.output_filename_prefix)
    net = paddle.jit.to_static(origin_model_new, input_spec=input_shape)
    paddle.jit.save(net, pruned_infer_model_dir)


def auto_model_forward(self,
                       input_ids,
                       token_type_ids=None,
                       position_ids=None,
                       attention_mask=[None, None]):
    """
    auto model forward
    """
    wtype = self.pooler.dense.fn.weight.dtype if hasattr(
        self.pooler.dense, 'fn') else self.pooler.dense.weight.dtype
    if attention_mask is None:
        attention_mask = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(wtype) * -1e9, axis=[1, 2])
    if attention_mask[0] is None:
        attention_mask[0] = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(wtype) * -1e9, axis=[1, 2])
    embedding_output = self.embeddings(input_ids=input_ids,
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
        idx = paddle.argsort(paddle.to_tensor(current_importance),
                             descending=True)
        nlp_utils.reorder_neuron(
            model.base_model.encoder.layers[layer].linear1.fn, idx, dim=1)

        nlp_utils.reorder_neuron(
            model.base_model.encoder.layers[layer].linear2.fn, idx, dim=0)


def compute_neuron_head_importance(model,
                                   data_loader,
                                   num_layers,
                                   num_heads,
                                   loss_fct,
                                   intermediate_name='linear1',
                                   output_name='linear2'):
    """
    Compute the importance of multi-head attention and feed-forward  neuron in
    each transformer layer.
    Args:
        model(paddle.nn.Layer):
            The instance of transformer model.
        data_loader (DataLoader):
            An iterable data loader is used for evaluate. An instance of
            `paddle.io.Dataloader`.
        num_layers (int):
            Number of transformer layers.
        num_heads (int):
            Number of heads in each multi-head attention.
        loss_fct (Loss|optional):
            Loss function can be a `paddle.nn.Layer` instance. Default: `nn.loss.CrossEntropyLoss()`.
        intermediate_name (str|optional):
            The name of intermediate `Linear` layer in feed-forward.
            Defaults to `linear1`.
        output_name (str|optional):
            The name of output `Linear` layer in feed-forward.
            Defaults to `linear2`.
    """
    head_importance = paddle.zeros(shape=[num_layers, num_heads],
                                   dtype='float32')
    head_mask = paddle.ones(shape=[num_layers, num_heads], dtype='float32')
    head_mask.stop_gradient = False

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []

    for name, w in model.named_parameters():
        if intermediate_name in name:
            if len(w.shape) > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if output_name in name:
            if len(w.shape) > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(np.zeros(shape=[w.shape[1]], dtype='float32'))

    data_loader = (data_loader, )
    for data in data_loader:
        for batch in data:
            if isinstance(batch, dict):
                input_ids, segment_ids, labels = batch['input_ids'], batch[
                    'token_type_ids'], batch['labels']
            else:
                input_ids, segment_ids, labels = batch
            logits = model(input_ids,
                           segment_ids,
                           attention_mask=[None, head_mask])
            loss = loss_fct(logits, labels)
            loss.backward()
            head_importance += paddle.abs(paddle.to_tensor(
                head_mask.gradient()))

            for w1, b1, w2, current_importance in zip(intermediate_weight,
                                                      intermediate_bias,
                                                      output_weight,
                                                      neuron_importance):
                current_importance += np.abs(
                    (np.sum(w1.numpy() * w1.gradient(), axis=0) +
                     b1.numpy() * b1.gradient()))
                current_importance += np.abs(
                    np.sum(w2.numpy() * w2.gradient(), axis=1))

    return head_importance, neuron_importance


def soft_cross_entropy(inp, target):
    """
    soft label cross entropy
    """
    inp_likelihood = F.log_softmax(inp, axis=-1)
    target_prob = F.softmax(target, axis=-1)
    return -1. * paddle.mean(paddle.sum(inp_likelihood * target_prob, axis=-1))


Trainer.prune = prune

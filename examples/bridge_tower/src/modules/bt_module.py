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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from visualdl import LogWriter

from paddlenlp.trainer.training_args import default_logdir
from paddlenlp.transformers import BertConfig, BertModel, RobertaConfig, RobertaModel

# import paddle
# import paddle.nn as nn
# import pypaddle_lightning as pl
# import paddle.nn.functional as F
# from .bert_model import BertConfig, BertModel, BertCrossLayer
# from . import swin_transformer as swin
# from . import vit_model as vit
# from .vit_model import resize_pos_embed
# flake8: noqa
from . import heads, meter_utils, objectives
from .bert_model import get_extended_attention_mask
from .clip_model import adapt_position_encoding, build_model
from .transformer import TransformerDecoderLayer


class BTTransformer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        # self.save_hyperparameters()
        self.hparams = config
        self.prepare_data_per_node = False
        self.is_clip = "clip" in config["vit"]
        self.is_swin = "swin" in config["vit"]
        self.is_vit = "vit" in config["vit"]
        self.jump_val_first_for_irtr_itm_irc = True
        # config["drop_rate"]=0.0 #test
        if "roberta" in config["tokenizer"]:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
                pad_token_id=1,
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after = config["image_size"]
        self.cross_modal_text_transform = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform = nn.Linear(config["input_image_embed_size"], config["hidden_size"])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if paddle.distributed.is_initialized():
            if paddle.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(
                        config["vit"],
                        resolution_after=resolution_after,
                        model_type=config["model_type"],
                        vit_layernorm_shared=config["vit_layernorm_shared"],
                        vit_remove_last=config["vit_remove_last"],
                    )
                elif self.is_swin:
                    getattr(swin, config["vit"])(
                        pretrained=True,
                        config=config,
                    )
                else:
                    getattr(vit, config["vit"])(
                        pretrained=True,
                        img_size=resolution_after,
                        model_type=config["model_type"],
                    )

                if "roberta" in config["tokenizer"]:
                    RobertaModel.from_pretrained(config["tokenizer"])
                else:
                    BertModel.from_pretrained(config["tokenizer"])

            paddle.distributed.barrier()

        if self.is_clip:
            self.vit_model = build_model(
                config["vit"],
                resolution_after=resolution_after,
                model_type=config["model_type"],
                vit_layernorm_shared=config["vit_layernorm_shared"],
                vit_remove_last=config["vit_remove_last"],
            )
        elif self.is_swin:
            self.vit_model = getattr(swin, config["vit"])(
                pretrained=True,
                config=config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        else:
            self.vit_model = getattr(vit, config["vit"])(
                pretrained=True,
                img_size=resolution_after,
                model_type=config["model_type"],
            )

        if "roberta" in config["tokenizer"]:
            self.text_transformer = RobertaModel.from_pretrained(
                config["tokenizer"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            self.text_transformer = BertModel.from_pretrained(config["tokenizer"])

        if not config["vit_layernorm_shared"] and config["vit_layernorm_init_from_vit"]:
            for ln in self.vit_model.visual.cross_modal_ln_separate:
                ln.weight.set_value(self.vit_model.visual.ln_post.weight)
                ln.bias.set_value(self.vit_model.visual.ln_post.bias)

        # BertCrossLayer is transformer decoder
        # self.cross_modal_image_layers = nn.LayerList([BertCrossLayer(bert_config) for _ in range(config['num_layers'])])
        self.cross_modal_image_layers = nn.LayerList(
            [
                TransformerDecoderLayer(
                    d_model=bert_config.hidden_size,
                    nhead=bert_config.num_attention_heads,
                    dim_feedforward=bert_config.intermediate_size,
                    activation=bert_config.hidden_act,
                    dropout=bert_config.hidden_dropout_prob,
                    attn_dropout=bert_config.attention_probs_dropout_prob,
                    normalize_before=False,
                )
                for _ in range(config["num_layers"])
            ]
        )
        # breakpoint()
        self.cross_modal_image_layers.apply(objectives.init_weights)
        # self.cross_modal_text_layers = nn.LayerList([BertCrossLayer(bert_config) for _ in range(config['num_layers'])])
        self.cross_modal_text_layers = nn.LayerList(
            [
                TransformerDecoderLayer(
                    d_model=bert_config.hidden_size,
                    nhead=bert_config.num_attention_heads,
                    dim_feedforward=bert_config.intermediate_size,
                    activation=bert_config.hidden_act,
                    dropout=bert_config.hidden_dropout_prob,
                    attn_dropout=bert_config.attention_probs_dropout_prob,
                    normalize_before=False,
                )
                for _ in range(config["num_layers"])
            ]
        )
        self.cross_modal_text_layers.apply(objectives.init_weights)

        # Class token => Linear => Tanh
        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        # Temperature for image text contrastive learning
        # config['temperature']
        self.temperature = paddle.create_parameter(
            shape=[1], dtype="float32", default_initializer=nn.initializer.Assign([config["temperature"]])
        )

        if config["loss_names"]["mlm"] > 0:
            # MLM Head weights don't tie with BERT Embedding weights. Train from scratch.
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if (
            config["loss_names"]["itm"] > 0
            or config["loss_names"]["itm_itc"] > 0
            or config["loss_names"]["irtr_itm_itc"] > 0
        ):
            self.itm_score = heads.ITMHead(config["hidden_size"] * 2)
            self.itm_score.apply(objectives.init_weights)

        if (
            config["loss_names"]["itc"] > 0
            or config["loss_names"]["itm_itc"] > 0
            or config["loss_names"]["irtr_itm_itc"] > 0
        ):
            self.itc_text_head = heads.ITCHead(config["hidden_size"], config["contrastive_hidden_size"])
            self.itc_text_head.apply(objectives.init_weights)
            self.itc_image_head = heads.ITCHead(config["hidden_size"], config["contrastive_hidden_size"])
            self.itc_image_head.apply(objectives.init_weights)

        hs = config["hidden_size"]

        # ===================== Initialize BT Components ===================== #
        # just for first layer
        self.cross_modal_text_layernorm = nn.LayerNorm(config["hidden_size"])
        self.cross_modal_text_layernorm.apply(objectives.init_weights)
        self.cross_modal_image_layernorm = nn.LayerNorm(config["hidden_size"])
        self.cross_modal_image_layernorm.apply(objectives.init_weights)

        self.cross_modal_text_link_tower = nn.LayerList(
            [heads.LinkTower(config) for _ in range(config["num_layers"] - 1)]
        )
        self.cross_modal_image_link_tower = nn.LayerList(
            [heads.LinkTower(config) for _ in range(config["num_layers"] - 1)]
        )

        self.cross_modal_text_link_tower.apply(objectives.init_weights)
        self.cross_modal_image_link_tower.apply(objectives.init_weights)

        # ===================== Load Pretrained METER Weights =====================

        if config["load_path"] != "" and not config["test_only"]:
            ckpt = paddle.load(config["load_path"])
            state_dict = ckpt
            if self.is_clip:
                # torch转换成paddle以后key变了，需要传入suffix
                state_dict = adapt_position_encoding(
                    state_dict,
                    after=resolution_after,
                    patch_size=config["patch_size"],
                    suffix="vit_model.vision_model.positional_embedding.weight",
                )
            elif self.is_swin:
                state_dict = swin_adapt_position_encoding(
                    state_dict, after=resolution_after, before=config["resolution_before"]
                )
            else:
                state_dict["vit_model.pos_embed"] = resize_pos_embed(
                    state_dict["vit_model.pos_embed"],
                    self.vit_model.pos_embed,
                    getattr(self.vit_model, "num_tokens", 1),
                    self.vit_model.patch_embed.grid_size,
                )

            self.load_dict(state_dict)

        # ===================== Downstream ===================== #

        hscale = config["head_hidden_scale"]
        if config["loss_names"]["vqa"] > 0:
            vs = config["vqav2_label_size"]
            if config["task_head_layers"] == 1:
                self.vqa_classifier = nn.Sequential(
                    nn.Linear(hs * 2, vs),
                )
            elif config["task_head_layers"] == 2:
                self.vqa_classifier = nn.Sequential(
                    nn.Linear(hs * 2, hs * 2 * hscale),
                    nn.LayerNorm(hs * 2 * hscale),
                    nn.GELU(),
                    nn.Linear(hs * 2 * hscale, vs),
                )
            self.vqa_classifier.apply(objectives.init_weights)

        if config["loss_names"]["snli"] > 0:
            if config["task_head_layers"] == 1:
                self.snli_classifier = nn.Sequential(
                    nn.Linear(hs * 2, 3),
                )
            elif config["task_head_layers"] == 2:
                self.snli_classifier = nn.Sequential(
                    nn.Linear(hs * 2, hs * 2 * hscale),
                    nn.LayerNorm(hs * 2 * hscale),
                    nn.GELU(),
                    nn.Linear(hs * 2 * hscale, 3),
                )
            self.snli_classifier.apply(objectives.init_weights)

        # ===================== load downstream (test_only) ======================

        if config["load_path"] != "" and config["test_only"]:
            ckpt = paddle.load(config["load_path"])
            state_dict = ckpt
            if self.is_clip:
                state_dict = adapt_position_encoding(
                    state_dict, after=resolution_after, patch_size=config["patch_size"]
                )
            elif self.is_swin:
                state_dict = swin_adapt_position_encoding(
                    state_dict, after=resolution_after, before=config["resolution_before"]
                )
            else:
                state_dict["vit_model.pos_embed"] = resize_pos_embed(
                    state_dict["vit_model.pos_embed"],
                    self.vit_model.pos_embed,
                    getattr(self.vit_model, "num_tokens", 1),
                    self.vit_model.patch_embed.grid_size,
                )
            self.load_state_dict(state_dict, strict=False)

        meter_utils.set_metrics(self)
        self.current_tasks = list()
        self.global_step = 0

        log_dir = default_logdir()
        self._LogWriter = LogWriter
        self.vdl_writer = self._LogWriter(logdir=log_dir)

    def get_cls_feats(self, text_feats, image_feats):
        # breakpoint()
        cls_feats_text = self.cross_modal_text_pooler(text_feats)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(image_feats)
        elif self.is_swin:
            avg_image_feats = self.avgpool(image_feats.transpose([1, 2])).reshape([image_feats.shape[0], 1, -1])
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        else:
            cls_feats_image = self.cross_modal_image_pooler(image_feats)
        return paddle.concat([cls_feats_text, cls_feats_image], axis=-1)

    def get_uni_modal_features(self, batch, fusion_features=False, itc=False):
        img = batch["image"][0]
        text_ids = batch[f"text_ids"]
        text_labels = batch[f"text_labels"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        input_shape = text_masks.shape
        # extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, self.device)
        extend_text_masks = get_extended_attention_mask(text_masks, input_shape)

        text_embedss = []
        split_index = len(self.text_transformer.encoder.layer) - self.hparams["num_layers"]
        index = 0
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
            index += 1
            if index > split_index:
                text_embedss.append(text_embeds)
        text_embedss = paddle.stack(text_embedss, dim=0)

        image_embedss = self.vit_model(img)
        image_embedss = image_embedss[len(image_embedss) - self.hparams["num_layers"] :]

        if itc:
            unimodal_feats_text = F.normalize(self.itc_text_head(text_embedss[-1][:, 0, :]), dim=-1, p=2)
            unimodal_feats_image = F.normalize(self.itc_image_head(image_embedss[-1][:, 0, :]), dim=-1, p=2)
            if not fusion_features:
                ret = {
                    "unimodal_feats_text": unimodal_feats_text,
                    "unimodal_feats_image": unimodal_feats_image,
                }
                return ret

        # cross_modal transform
        text_embedss = self.cross_modal_text_transform(text_embedss)
        image_embedss = self.cross_modal_image_transform(image_embedss)

        if not itc:
            ret = {
                "extend_text_masks": extend_text_masks,
                "text_embedss": text_embedss,
                "image_embedss": image_embedss,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
            }
        else:
            if fusion_features:
                ret = {
                    "unimodal_feats_text": unimodal_feats_text,
                    "unimodal_feats_image": unimodal_feats_image,
                    "extend_text_masks": extend_text_masks,
                    "text_embedss": text_embedss,
                    "image_embedss": image_embedss,
                    "text_labels": text_labels,
                    "text_ids": text_ids,
                    "text_masks": text_masks,
                }
        return ret

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        irtr_len_text=0,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        input_shape = text_masks.shape
        # extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, self.device)
        extend_text_masks = get_extended_attention_mask(text_masks, input_shape)

        split_index = len(self.text_transformer.encoder.layers) - self.hparams["num_layers"] + 1
        # 这部分参数有微小差别
        for layer in self.text_transformer.encoder.layers[:split_index]:
            # get hidden_states
            text_embeds = layer(text_embeds, src_mask=extend_text_masks)

        if self.is_clip:
            # image_embeds = self.vit_model.vision_model.forward_pre(img.type(self.vit_model.dtype))
            image_embeds = self.vit_model.vision_model.forward_pre(img)
            # resblocks
            # transformer input is [batch_size, seq_len, embedding_dim]
            # image_embeds = image_embeds.transpose([1, 0, 2])
            for block in self.vit_model.vision_model.transformer.layers[:split_index]:
                image_embeds = block(image_embeds)
            # image_embeds = image_embeds.transpose([1, 0, 2])
            image_embeds_ = self.vit_model.vision_model.forward_post(image_embeds)
            # breakpoint()
        else:
            image_embeds = self.vit_model.forward_pre(img)
            for block in self.vit_model.blocks[:split_index]:
                image_embeds = block(image_embeds)
            image_embeds_ = self.vit_model.forward_post(image_embeds)

        if self.hparams["num_layers"] == 0:
            cls_feats = self.get_cls_feats(text_embeds, image_embeds_)

            ret = {
                "text_feats": text_embeds,
                "image_feats": image_embeds_,
                "cls_feats": cls_feats,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
            }
            return ret

        # first layer
        x = self.cross_modal_text_transform(text_embeds)
        text_token_type_embeddings = self.token_type_embeddings(paddle.zeros(shape=[1], dtype="int64")).expand_as(x)
        x = self.cross_modal_text_layernorm(x + text_token_type_embeddings)

        image_embeds_ = self.cross_modal_image_transform(image_embeds_)
        image_token_type_embeddings = self.token_type_embeddings(
            paddle.zeros(shape=[1], dtype="int64").fill_(image_token_type_idx)
        ).expand_as(image_embeds_)
        image_embeds_ = image_embeds_ + image_token_type_embeddings
        y = self.cross_modal_image_layernorm(image_embeds_)
        if irtr_len_text > 0:
            _bs, _L, _D = image_embeds_.shape
            y = y.unsqueeze(1).expand(_bs, irtr_len_text, _L, _D).reshape([-1, _L, _D])
        image_masks = paddle.ones((y.shape[0], y.shape[1]), dtype="int64")
        # extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.shape, self.device)
        extend_image_masks = get_extended_attention_mask(image_masks, image_masks.shape)
        x1 = self.cross_modal_text_layers[0](x, y, extend_text_masks, extend_image_masks)
        y1 = self.cross_modal_image_layers[0](y, x, extend_image_masks, extend_text_masks)

        link_layer_index = 0

        # link tower fusion
        for i in range(split_index, len(self.text_transformer.encoder.layers)):
            text_embeds = self.text_transformer.encoder.layers[i](text_embeds, extend_text_masks)

            if self.is_clip:
                image_embeds = self.vit_model.vision_model.transformer.layers[i](image_embeds)
                image_embeds_ = (
                    self.cross_modal_image_transform(self.vit_model.vision_model.forward_post(image_embeds))
                    + image_token_type_embeddings
                )
            else:
                image_embeds = self.vit_model.blocks[i](image_embeds)
                image_embeds_ = (
                    self.cross_modal_image_transform(self.vit_model.forward_post(image_embeds))
                    + image_token_type_embeddings
                )

            text_link_tower = self.cross_modal_text_link_tower[link_layer_index]
            image_link_tower = self.cross_modal_image_link_tower[link_layer_index]

            x1_ = text_link_tower(self.cross_modal_text_transform(text_embeds) + text_token_type_embeddings, x1)
            if irtr_len_text > 0:
                y1_ = image_link_tower(
                    image_embeds_.unsqueeze(1).expand(_bs, irtr_len_text, _L, _D).reshape([-1, _L, _D]), y1
                )
            else:
                y1_ = image_link_tower(image_embeds_, y1)
            # breakpoint()
            x1 = self.cross_modal_text_layers[link_layer_index + 1](x1_, y1_, extend_text_masks, extend_image_masks)
            y1 = self.cross_modal_image_layers[link_layer_index + 1](y1_, x1_, extend_image_masks, extend_text_masks)

            link_layer_index += 1

        text_feats, image_feats = x1, y1
        cls_feats = self.get_cls_feats(text_feats, image_feats)
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def forward(self, batch, split, loss_name, **kwargs):
        self.global_step = kwargs.get("global_step", 0)
        self.status = kwargs.get("status", None)
        self.eval_step = kwargs.get("eval_step", 0)

        if self.status is not None:
            if self.status == "validation_epoch_end":
                ret = meter_utils.epoch_wrapup(self, "val")
            elif self.status == "training_epoch_end":
                ret = meter_utils.epoch_wrapup(self, "train")
            # Update logs for every step
            for k, v in ret.items():
                if isinstance(v, (int, float)):
                    self.vdl_writer.add_scalar(k, v, self.global_step + self.eval_step)
            return ret

        ret = dict()
        if "snli" in loss_name:
            ret.update(objectives.compute_snli(self, batch, split))

        if "mlm" in loss_name:
            ret.update(objectives.compute_mlm(self, batch, split))
        if "itm" in loss_name:
            ret.update(objectives.compute_itm(self, batch, split))
        # 输出卡0的指标到visualdl
        if paddle.distributed.get_rank() == 0:
            # Update logs for every step
            for k, v in ret.items():
                if isinstance(v, (int, float)):
                    self.vdl_writer.add_scalar(k, v, self.global_step + self.eval_step)
        return ret

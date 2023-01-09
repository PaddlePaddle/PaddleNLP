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

import functools
import json

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange
from tqdm import tqdm


class DistributedSampler(paddle.io.DistributedBatchSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(
            dataset=dataset, batch_size=1, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last
        )


def init_weights(layer):
    """Initialization hook"""
    if isinstance(layer, (nn.Linear, nn.Embedding)):
        # only support dygraph, use truncated_normal and make it inplace
        # and configurable later
        layer.weight.set_value(
            paddle.tensor.normal(
                mean=0.0,
                std=0.02,
                shape=layer.weight.shape,
            )
        )
    elif isinstance(layer, nn.LayerNorm):
        layer.weight.set_value(paddle.ones_like(layer.weight))
        layer.bias.set_value(paddle.zeros_like(layer.bias))

    if isinstance(layer, nn.Linear) and layer.bias is not None:
        layer.bias.set_value(paddle.zeros_like(layer.bias))


# pre-train
def compute_mlm(model, batch, split):
    infer = model.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = model.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    logsoftmax = F.log_softmax(mlm_logits.reshape([-1, mlm_logits.shape[-1]]), axis=1)
    mlm_loss = F.nll_loss(logsoftmax, mlm_labels.reshape([-1]), ignore_index=-100)
    # There is a bug when using ignore_index
    # mlm_loss = F.cross_entropy(
    #     mlm_logits.reshape([-1, mlm_logits.shape[-1]]),
    #     mlm_labels.reshape([-1]),
    #     ignore_index=-100,
    # )
    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    loss_name = "mlm"
    loss = getattr(model, f"{split}_{loss_name}_loss")
    acc = getattr(model, f"{split}_{loss_name}_accuracy")

    loss.update(ret["mlm_loss"])
    result = acc.compute(ret["mlm_logits"].reshape([-1, mlm_logits.shape[-1]]), ret["mlm_labels"].reshape([-1]))

    acc.update(result)

    # 聚合各个节点的loss
    example_train_loss = loss.accumulate()
    dist.all_reduce(example_train_loss)
    example_train_loss = example_train_loss / dist.get_world_size()
    metric_dict = {
        f"{split}/{loss_name}/loss": example_train_loss.item(),
        f"{split}/{loss_name}/accuracy": acc.accumulate(),
    }
    ret.update(metric_dict)
    return ret


def compute_itm(model, batch, split):

    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = paddle.concat([paddle.ones([pos_len]), paddle.zeros([neg_len])])
    itm_labels = itm_labels[paddle.randperm(itm_labels.shape[0])]
    itm_images = [
        paddle.stack([ti if itm_labels[i] == 1 else fi for i, (ti, fi) in enumerate(zip(bti, bfi))])
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = model.infer(batch)

    itm_logits = model.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.astype("int64"))

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    loss_name = "itm"

    loss = getattr(model, f"{split}_{loss_name}_loss")
    acc = getattr(model, f"{split}_{loss_name}_accuracy")

    loss.update(ret["itm_loss"])
    result = acc.compute(ret["itm_logits"], ret["itm_labels"])

    acc.update(result)
    loss_name = "itm"
    # 聚合各个节点的loss
    example_train_loss = loss.accumulate()
    dist.all_reduce(example_train_loss)
    example_train_loss = example_train_loss / dist.get_world_size()
    metric_dict = {
        f"{split}/{loss_name}/loss": example_train_loss.item(),
        f"{split}/{loss_name}/accuracy": acc.accumulate(),
    }
    ret.update(metric_dict)
    return ret


def compute_itc(pl_module, batch, split):
    assert batch["image"][0].size(0) == len(batch["text"])
    bs, rank = len(batch["text"]), paddle.distributed.get_rank()

    with paddle.no_grad():
        pl_module.temperature.clamp_(0.001, 0.5)

    infer = pl_module.get_uni_modal_features(batch, itc=True)
    unimodal_feats_text = infer["unimodal_feats_text"]
    unimodal_feats_image = infer["unimodal_feats_image"]

    if pl_module.hparams.config["gather_with_grads"]:
        gather_unimodal_feats_text = pl_module.all_gather(unimodal_feats_text, sync_grads=True)
        gather_unimodal_feats_image = pl_module.all_gather(unimodal_feats_image, sync_grads=True)
    else:
        with paddle.no_grad():
            gather_unimodal_feats_text = pl_module.all_gather(unimodal_feats_text)
            gather_unimodal_feats_image = pl_module.all_gather(unimodal_feats_image)
        gather_unimodal_feats_text[rank] = unimodal_feats_text
        gather_unimodal_feats_image[rank] = unimodal_feats_image

    gather_unimodal_feats_text = gather_unimodal_feats_text.view((-1,) + (gather_unimodal_feats_text.shape)[2:])
    gather_unimodal_feats_image = gather_unimodal_feats_image.view((-1,) + (gather_unimodal_feats_image.shape)[2:])

    logit_scale = paddle.log(1 / pl_module.temperature).exp()
    itc_logits_i2t = logit_scale * unimodal_feats_image @ gather_unimodal_feats_text.t()
    itc_logits_t2i = logit_scale * unimodal_feats_text @ gather_unimodal_feats_image.t()

    itc_labels = paddle.arange(bs).to(pl_module.device)
    itc_labels = itc_labels + bs * rank
    i2t_loss = F.cross_entropy(itc_logits_i2t, itc_labels)
    t2i_loss = F.cross_entropy(itc_logits_t2i, itc_labels)
    itc_loss = (i2t_loss + t2i_loss) / 2

    ret = {
        "itc_loss": itc_loss,
    }

    loss_name = "itc"

    if pl_module.hparams.config["num_layers"] == 0:
        loss_name = "irtr_itm"

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["itc_loss"])
    pl_module.log(f"{split}/{loss_name}/loss", loss)

    return ret


def compute_itm_itc(pl_module, batch, split, pretrain=False):
    # REMEMBER: No need to draw false images for image text matching in data preprocessing.
    assert batch["image"][0].size(0) == len(batch["text"])
    bs, rank = len(batch["text"]), paddle.distributed.get_rank()

    # forward the positive image-text pair
    with paddle.no_grad():
        pl_module.temperature.clamp_(0.001, 0.5)

    infer = pl_module.get_uni_modal_features(batch, fusion_features=True, itc=True)
    unimodal_feats_text = infer["unimodal_feats_text"]
    unimodal_feats_image = infer["unimodal_feats_image"]

    if pl_module.hparams.config["gather_with_grads"]:
        gather_unimodal_feats_text = pl_module.all_gather(unimodal_feats_text, sync_grads=True)
        gather_unimodal_feats_image = pl_module.all_gather(unimodal_feats_image, sync_grads=True)
    else:
        with paddle.no_grad():
            gather_unimodal_feats_text = pl_module.all_gather(unimodal_feats_text)
            gather_unimodal_feats_image = pl_module.all_gather(unimodal_feats_image)
        gather_unimodal_feats_text[rank] = unimodal_feats_text
        gather_unimodal_feats_image[rank] = unimodal_feats_image

    gather_unimodal_feats_text = gather_unimodal_feats_text.view((-1,) + (gather_unimodal_feats_text.shape)[2:])
    gather_unimodal_feats_image = gather_unimodal_feats_image.view((-1,) + (gather_unimodal_feats_image.shape)[2:])

    logit_scale = paddle.log(1 / pl_module.temperature).exp()
    itc_logits_i2t = logit_scale * unimodal_feats_image @ gather_unimodal_feats_text.t()
    itc_logits_t2i = logit_scale * unimodal_feats_text @ gather_unimodal_feats_image.t()

    if pretrain:
        itc_labels = paddle.arange(bs).to(pl_module.device)
        itc_labels = itc_labels + bs * rank
    else:
        idx = paddle.LongTensor(batch["img_index"]).view(-1, 1).to(pl_module.device)
        idx_all = pl_module.all_gather(idx).view(-1, 1)
        assert idx_all.size(0) == gather_unimodal_feats_image.size(0)
        idx_all = paddle.eq(idx_all, idx_all.t()).to(pl_module.device)
        idx_all = idx_all[bs * rank : bs * (rank + 1)]
        pos_idx = idx_all.float()
        assert pos_idx.size(0) == len(idx)
        itc_labels = pos_idx / pos_idx.sum(1, keepdim=True)

    i2t_loss = F.cross_entropy(itc_logits_i2t, itc_labels)
    t2i_loss = F.cross_entropy(itc_logits_t2i, itc_labels)
    itc_loss = (i2t_loss + t2i_loss) / 2

    if pretrain:
        loss_name = "itc"
    else:
        loss_name = "irtr_itc"

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(itc_loss)
    pl_module.log(f"{split}/{loss_name}/loss", loss)

    # sample hard negative images for image text matching from image text contrastive logits
    if pl_module.hparams.config["gather_global_negative"]:
        # select a negative image for each text
        with paddle.no_grad():
            weights_i2t = F.softmax(itc_logits_i2t, dim=-1)
            weights_t2i = F.softmax(itc_logits_t2i, dim=-1)
            if pretrain:
                weights_i2t[:, bs * rank : bs * (rank + 1)].fill_diagonal_(0)
                weights_t2i[:, bs * rank : bs * (rank + 1)].fill_diagonal_(0)
            else:
                weights_i2t.masked_fill_(idx_all, 0)
                weights_t2i.masked_fill_(idx_all, 0)

        global_image_embedss = pl_module.all_gather(infer["image_embedss"].transpose(0, 1), sync_grads=True).view(
            -1, infer["image_embedss"].size(0), infer["image_embedss"].size(2), infer["image_embedss"].size(3)
        )

        image_embeds_neg = []
        for b in range(bs):
            try:
                neg_idx = paddle.multinomial(weights_t2i[b], 1).item()
            except Exception:
                neg_idx = paddle.multinomial(weights_t2i[b] + 1e-5, 1).item()
            image_embeds_neg.append(global_image_embedss[neg_idx])
        image_embeds_neg = paddle.stack(image_embeds_neg, dim=1)
        # del global_image_embedss

        # select a negative text for each image
        global_text_embedss = pl_module.all_gather(infer["text_embedss"].transpose(0, 1), sync_grads=True).view(
            -1, infer["text_embedss"].size(0), infer["text_embedss"].size(2), infer["text_embedss"].size(3)
        )
        global_text_masks = pl_module.all_gather(infer["text_masks"]).view(-1, infer["text_masks"].size(1))

        text_embeds_neg = []
        text_masks_neg = []
        for b in range(bs):
            try:
                neg_idx = paddle.multinomial(weights_i2t[b], 1).item()
            except Exception:
                neg_idx = paddle.multinomial(weights_i2t[b] + 1e-5, 1).item()
            text_embeds_neg.append(global_text_embedss[neg_idx])
            text_masks_neg.append(global_text_masks[neg_idx])
        text_embeds_neg = paddle.stack(text_embeds_neg, dim=1)
        text_masks_neg = paddle.stack(text_masks_neg, dim=0)
        # del global_text_embedss, global_text_masks
    else:
        # select a negative image for each text
        with paddle.no_grad():
            weights_i2t = F.softmax(itc_logits_i2t[:, bs * rank : bs * (rank + 1)], dim=-1)
            weights_t2i = F.softmax(itc_logits_t2i[:, bs * rank : bs * (rank + 1)], dim=-1)
            if pretrain:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                mask = paddle.eq(idx, idx.t()).to(pl_module.device)
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        for b in range(bs):
            try:
                neg_idx = paddle.multinomial(weights_t2i[b], 1).item()
            except Exception:
                neg_idx = paddle.multinomial(weights_t2i[b] + 1e-5, 1).item()
            image_embeds_neg.append(infer["image_embedss"][:, neg_idx])
        image_embeds_neg = paddle.stack(image_embeds_neg, dim=1)

        # select a negative text for each image
        text_embeds_neg = []
        text_masks_neg = []
        for b in range(bs):
            try:
                neg_idx = paddle.multinomial(weights_i2t[b], 1).item()
            except Exception:
                neg_idx = paddle.multinomial(weights_i2t[b] + 1e-5, 1).item()
            text_embeds_neg.append(infer["text_embedss"][:, neg_idx])
            text_masks_neg.append(infer["text_masks"][neg_idx])
        text_embeds_neg = paddle.stack(text_embeds_neg, dim=1)
        text_masks_neg = paddle.stack(text_masks_neg, dim=0)

    # pack the negative image-text pairs for fusion, which is 2 x batch_size
    text_embedss = paddle.cat([infer["text_embedss"], text_embeds_neg], dim=1)
    text_masks = paddle.cat([infer["text_masks"], text_masks_neg], dim=0)
    extend_text_masks = pl_module.text_transformer.get_extended_attention_mask(
        text_masks, text_masks.size(), pl_module.device
    )

    image_embedss = paddle.cat([image_embeds_neg, infer["image_embedss"]], dim=1)

    # fusion
    pos_cls_feats = pl_module.infer_fusion(infer["image_embedss"], infer["text_embedss"], infer["extend_text_masks"])[
        "cls_feats"
    ]
    neg_cls_feats = pl_module.infer_fusion(image_embedss, text_embedss, extend_text_masks)["cls_feats"]
    cls_feats = paddle.cat([pos_cls_feats, neg_cls_feats], dim=0)

    itm_labels = paddle.cat([paddle.ones(bs, dtype=paddle.long), paddle.zeros(2 * bs, dtype=paddle.long)]).to(
        pl_module.device
    )

    itm_logits = pl_module.itm_score(cls_feats)
    itm_loss = F.cross_entropy(itm_logits, itm_labels)

    ret = {
        "itc_loss": itc_loss,
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    if pretrain:
        loss_name = "itm"
    else:
        loss_name = "irtr_itm"
    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    return ret


def compute_itm_itc_meter(pl_module, batch, split, pretrain=False):
    # REMEMBER: No need to draw false images for image text matching in data preprocessing.
    assert batch["image"][0].size(0) == len(batch["text"])
    bs, rank = len(batch["text"]), paddle.distributed.get_rank()

    # forward the positive image-text pair
    with paddle.no_grad():
        pl_module.temperature.clamp_(0.001, 0.5)

    infer = pl_module.get_uni_modal_features(batch, fusion_features=True, itc=True)
    unimodal_feats_text = infer["unimodal_feats_text"]
    unimodal_feats_image = infer["unimodal_feats_image"]

    if pl_module.hparams.config["gather_with_grads"]:
        gather_unimodal_feats_text = pl_module.all_gather(unimodal_feats_text, sync_grads=True)
        gather_unimodal_feats_image = pl_module.all_gather(unimodal_feats_image, sync_grads=True)
    else:
        with paddle.no_grad():
            gather_unimodal_feats_text = pl_module.all_gather(unimodal_feats_text)
            gather_unimodal_feats_image = pl_module.all_gather(unimodal_feats_image)
        gather_unimodal_feats_text[rank] = unimodal_feats_text
        gather_unimodal_feats_image[rank] = unimodal_feats_image

    gather_unimodal_feats_text = gather_unimodal_feats_text.view((-1,) + (gather_unimodal_feats_text.shape)[2:])
    gather_unimodal_feats_image = gather_unimodal_feats_image.view((-1,) + (gather_unimodal_feats_image.shape)[2:])

    logit_scale = paddle.log(1 / pl_module.temperature).exp()
    itc_logits_i2t = logit_scale * unimodal_feats_image @ gather_unimodal_feats_text.t()
    itc_logits_t2i = logit_scale * unimodal_feats_text @ gather_unimodal_feats_image.t()

    if pretrain:
        itc_labels = paddle.arange(bs).to(pl_module.device)
        itc_labels = itc_labels + bs * rank
    else:
        idx = paddle.LongTensor(batch["img_index"]).view(-1, 1).to(pl_module.device)
        idx_all = pl_module.all_gather(idx).view(-1, 1)
        assert idx_all.size(0) == gather_unimodal_feats_image.size(0)
        idx_all = paddle.eq(idx_all, idx_all.t()).to(pl_module.device)
        idx_all = idx_all[bs * rank : bs * (rank + 1)]
        pos_idx = idx_all.float()
        assert pos_idx.size(0) == len(idx)
        itc_labels = pos_idx / pos_idx.sum(1, keepdim=True)

    i2t_loss = F.cross_entropy(itc_logits_i2t, itc_labels)
    t2i_loss = F.cross_entropy(itc_logits_t2i, itc_labels)
    itc_loss = (i2t_loss + t2i_loss) / 2

    if pretrain:
        loss_name = "itc"
    else:
        loss_name = "irtr_itc"

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(itc_loss)
    pl_module.log(f"{split}/{loss_name}/loss", loss)

    # sample hard negative images for image text matching from image text contrastive logits
    # select a negative image for each text
    with paddle.no_grad():
        weights_i2t = F.softmax(itc_logits_i2t[:, bs * rank : bs * (rank + 1)], dim=-1)
        weights_t2i = F.softmax(itc_logits_t2i[:, bs * rank : bs * (rank + 1)], dim=-1)
        if pretrain:
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)
        else:
            mask = paddle.eq(idx, idx.t()).to(pl_module.device)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

    image_embeds_neg = []
    for b in range(bs):
        try:
            neg_idx = paddle.multinomial(weights_t2i[b], 1).item()
        except Exception:
            neg_idx = paddle.multinomial(weights_t2i[b] + 1e-5, 1).item()
        image_embeds_neg.append(infer["image_embeds"][neg_idx])
    image_embeds_neg = paddle.stack(image_embeds_neg, dim=0)

    # select a negative text for each image
    text_embeds_neg = []
    text_masks_neg = []
    for b in range(bs):
        try:
            neg_idx = paddle.multinomial(weights_i2t[b], 1).item()
        except Exception:
            neg_idx = paddle.multinomial(weights_i2t[b] + 1e-5, 1).item()
        text_embeds_neg.append(infer["text_embeds"][neg_idx])
        text_masks_neg.append(infer["text_masks"][neg_idx])
    text_embeds_neg = paddle.stack(text_embeds_neg, dim=0)
    text_masks_neg = paddle.stack(text_masks_neg, dim=0)

    # pack the negative image-text pairs for fusion, which is 2 x batch_size
    text_embeds = paddle.cat([infer["text_embeds"], text_embeds_neg], dim=0)
    text_masks = paddle.cat([infer["text_masks"], text_masks_neg], dim=0)
    extend_text_masks = pl_module.text_transformer.get_extended_attention_mask(
        text_masks, text_masks.size(), pl_module.device
    )

    image_embeds = paddle.cat([image_embeds_neg, infer["image_embeds"]], dim=0)

    # fusion
    pos_cls_feats = pl_module.infer_fusion(infer["image_embeds"], infer["text_embeds"], infer["extend_text_masks"])[
        "cls_feats"
    ]
    neg_cls_feats = pl_module.infer_fusion(image_embeds, text_embeds, extend_text_masks)["cls_feats"]
    cls_feats = paddle.cat([pos_cls_feats, neg_cls_feats], dim=0)

    itm_labels = paddle.cat([paddle.ones(bs, dtype=paddle.long), paddle.zeros(2 * bs, dtype=paddle.long)]).to(
        pl_module.device
    )

    itm_logits = pl_module.itm_score(cls_feats)
    itm_loss = F.cross_entropy(itm_logits, itm_labels)

    ret = {
        "itc_loss": itc_loss,
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    if pretrain:
        loss_name = "itm"
    else:
        loss_name = "irtr_itm"
    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    return ret


# fine-tune
def compute_snli(model, batch, split):
    infer = model.infer(batch)
    snli_logits = model.snli_classifier(infer["cls_feats"])

    snli_labels = batch["labels"]
    # snli_labels = paddle.to_tensor(snli_labels)
    snli_loss = F.cross_entropy(snli_logits, snli_labels.reshape([-1]))

    ret = {
        "snli_loss": snli_loss,
        "snli_logits": snli_logits,
        "snli_labels": snli_labels,
    }

    loss_name = "snli"
    if split == "train":
        loss_fn = getattr(model, f"{split}_{loss_name}_loss")
        acc_metric = getattr(model, f"{split}_{loss_name}_accuracy")
        result = acc_metric.compute(ret["snli_logits"], ret["snli_labels"])

        acc_metric.update(result)
        # 聚合各个节点的loss
        example_train_loss = loss_fn.accumulate()
        dist.all_reduce(example_train_loss)
        example_train_loss = example_train_loss / dist.get_world_size()
        # Get accumulated loss and accuracy
        example_train_loss = example_train_loss.item()
        accuracy = acc_metric.accumulate()

        metric_dict = {
            f"{split}/{loss_name}/loss": example_train_loss,
            f"{split}/{loss_name}/accuracy": accuracy,
        }
        ret.update(metric_dict)
    else:
        val_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if val_batches:
            val_loss = getattr(model, f"val_{loss_name}_loss")
            val_loss.update(F.cross_entropy(ret["snli_logits"][val_batches], ret["snli_labels"][val_batches]))
            val_acc = getattr(model, f"val_{loss_name}_accuracy")
            result = val_acc.compute(ret["snli_logits"][val_batches], ret["snli_labels"][val_batches])
            val_acc.update(result)
            # pl_module.log(f"val/snli/loss", val_loss)
            # pl_module.log(f"val/snli/accuracy", val_acc)
            example_val_loss = val_loss.accumulate()
        else:
            val_loss = getattr(model, f"val_{loss_name}_loss")
            example_val_loss = val_loss.accumulate()
            val_acc = getattr(model, f"val_{loss_name}_accuracy")

        # 聚合各个节点的loss
        dist.all_reduce(example_val_loss)
        example_val_loss = example_val_loss / dist.get_world_size()
        # Get accumulated loss and accuracy
        example_val_loss = example_val_loss.item()
        val_acc = val_acc.accumulate()
        metric_dict = {"val/snli/loss": example_val_loss, "val/snli/accuracy": val_acc}
        ret.update(metric_dict)

        if test_batches:
            test_loss = getattr(model, f"test_{loss_name}_loss")

            test_loss.update(F.cross_entropy(ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]))
            test_acc = getattr(model, f"test_{loss_name}_accuracy")

            result = test_acc.compute(ret["snli_logits"][test_batches], ret["snli_labels"][test_batches])
            test_acc.update(result)

            # metric_dict = {f"test/{loss_name}/loss": test_loss.accumulate().item(),
            #             f"test/{loss_name}/accuracy": test_acc.accumulate()}
            example_test_loss = test_loss.accumulate()
            # pl_module.log(f"test/snli/loss", test_loss)
            # pl_module.log(f"test/snli/accuracy", test_acc)
        else:
            test_loss = getattr(model, f"test_{loss_name}_loss")
            example_test_loss = test_loss.accumulate()
            test_acc = getattr(model, f"test_{loss_name}_accuracy")

        # 聚合各个节点的loss
        dist.all_reduce(example_test_loss)
        example_test_loss = example_test_loss / dist.get_world_size()
        # Get accumulated loss and accuracy
        example_test_loss = example_test_loss.item()
        test_acc = test_acc.accumulate()

        metric_dict = {
            f"test/{loss_name}/loss": example_test_loss,
            f"test/{loss_name}/accuracy": test_acc,
        }
        # model.log(metric_dict)
        # model.log(f"{split}/{loss_name}/accuracy", acc)
        # print(metric_dict)
        ret.update(metric_dict)

    return ret


def compute_vqa(model, batch, split):
    # infer = model.infer(batch)
    model_infer = model._layers if isinstance(model, paddle.DataParallel) else model

    infer = model_infer.infer(batch)
    vqa_logits = model_infer.vqa_classifier(infer["cls_feats"])
    # breakpoint()
    vqa_targets = paddle.zeros([len(vqa_logits), vqa_logits.shape[1]])

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }
    return ret


def compute_nlvr2(pl_module, batch, split):
    infer1 = pl_module.infer(batch, image_token_type_idx=1)
    infer2 = pl_module.infer(batch, image_token_type_idx=2)

    cls_feats = paddle.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    if pl_module.hparams.config["nlvr2_drop_rate"] > 0:
        cls_feats = pl_module.nlvr2_classifier_dropout(cls_feats)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = paddle.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    loss_name = "nlvr2"

    if split == "train":
        loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(ret["nlvr2_logits"], ret["nlvr2_labels"])
        pl_module.log(f"{split}/{loss_name}/loss", loss)
        pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    else:
        val_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if val_batches:
            val_loss = getattr(pl_module, f"val_{loss_name}_loss")(
                F.cross_entropy(ret["nlvr2_logits"][val_batches], ret["nlvr2_labels"][val_batches])
            )
            val_acc = getattr(pl_module, f"val_{loss_name}_accuracy")(
                ret["nlvr2_logits"][val_batches], ret["nlvr2_labels"][val_batches]
            )
            pl_module.log("val/nlvr2/loss", val_loss)
            pl_module.log("val/nlvr2/accuracy", val_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_{loss_name}_loss")(
                F.cross_entropy(ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches])
            )
            test_acc = getattr(pl_module, f"test_{loss_name}_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log("test/nlvr2/loss", test_loss)
            pl_module.log("test/nlvr2/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch, split):
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = paddle.stack([batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1)
    text_masks = paddle.stack([batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1)
    text_labels = paddle.stack([batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1)

    text_ids = paddle.stack([batch["text_ids"], text_ids], dim=1)
    text_masks = paddle.stack([batch["text_masks"], text_masks], dim=1)
    text_labels = paddle.stack([batch["text_labels"], text_labels], dim=1)

    infer = pl_module.infer(
        {
            "image": batch["image"],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        },
        irtr_len_text=false_len + 1,
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = paddle.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    loss_name = "irtr"

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["irtr_loss"])
    pl_module.log(f"{split}/{loss_name}/loss", loss)

    return ret


# calculate recall for irtr task
@paddle.no_grad()
def compute_irtr_recall(pl_module, split):
    print("[Evaluation] load irtr dataset for text features caching")
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_loader = paddle.utils.data.DataLoader(
        text_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_text"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=text_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    print("[Evaluation] load irtr dataset for image features caching")
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split, image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = paddle.utils.data.DataLoader(
        image_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_image"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=image_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    print("[Evaluation] start to cache the text features")
    text_embedss_cache, extend_text_masks_cache, tiids = list(), list(), list()
    for _b in tqdm(text_loader, desc="text prefetch loop"):
        text_embedss, extend_text_masks = pl_module.infer_text(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
            },
        )
        text_embedss_cache.append(text_embedss)
        extend_text_masks_cache.append(extend_text_masks)
        tiids += _b["img_index"]

    text_embedss_cache = paddle.cat(text_embedss_cache, dim=1)
    extend_text_masks_cache = paddle.cat(extend_text_masks_cache, dim=0)
    tiids = paddle.LongTensor(tiids)

    # gather all text features
    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()

    text_embedss_cache = (
        pl_module.all_gather(text_embedss_cache.transpose(0, 1))
        .to(pl_module.device)
        .view(-1, text_embedss_cache.size(0), text_embedss_cache.size(2), text_embedss_cache.size(3))
        .transpose(0, 1)
    )
    extend_text_masks_cache = (
        pl_module.all_gather(extend_text_masks_cache)
        .to(pl_module.device)
        .view(-1, extend_text_masks_cache.size(1), extend_text_masks_cache.size(2), extend_text_masks_cache.size(3))
    )
    tiids = pl_module.all_gather(tiids).to(pl_module.device).view(-1)

    print("[Evaluation] start to cache the image features")
    image_embedss_cache, iids_cache = list(), list()
    for _b in tqdm(image_loader, desc="image prefetch loop"):
        image_embedss = pl_module.infer_image(img=_b["image"][0].to(pl_module.device))
        image_embedss_cache.append(image_embedss)
        iids_cache += _b["img_index"]
    image_embedss_cache = paddle.cat(image_embedss_cache, dim=1)

    image_index, rank_scores, rank_iids = 0, list(), list()

    text_chunk_size = pl_module.hparams.config["per_gpu_eval_batchsize_fusion_text"]
    if text_embedss_cache.size(1) % text_chunk_size == 0:
        text_chunk_num = text_embedss_cache.size(1) // text_chunk_size
    else:
        text_chunk_num = text_embedss_cache.size(1) // text_chunk_size + 1

    print("[Evaluation] start to compute the irtr recall")
    for _iid in tqdm(iids_cache, desc="rank loop"):
        image_embedss = image_embedss_cache[:, image_index]
        image_index += 1

        img_batch_score = list()
        for _i in range(text_chunk_num):
            text_embedss = text_embedss_cache[:, _i * text_chunk_size : (_i + 1) * text_chunk_size]
            extend_text_masks = extend_text_masks_cache[_i * text_chunk_size : (_i + 1) * text_chunk_size]
            if pl_module.hparams.config["amp_flag"]:
                with paddle.cuda.amp.autocast():
                    score = pl_module.rank_output(
                        pl_module.infer_fusion(
                            image_embedss,
                            text_embedss,
                            extend_text_masks,
                            irtr_len_image=text_embedss.size(1),
                        )["cls_feats"]
                    )[:, 0]
            else:
                score = pl_module.rank_output(
                    pl_module.infer_fusion(
                        image_embedss,
                        text_embedss,
                        extend_text_masks,
                        irtr_len_image=text_embedss.size(1),
                    )["cls_feats"]
                )[:, 0]
            img_batch_score.append(score)

        img_batch_score = paddle.cat(img_batch_score)
        rank_scores.append(img_batch_score)
        rank_iids.append(_iid)
    rank_iids = paddle.LongTensor(rank_iids)
    rank_scores = paddle.cat(rank_scores, dim=0)

    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()
    iids = pl_module.all_gather(rank_iids).to(pl_module.device).view(-1)
    scores = pl_module.all_gather(rank_scores).to(pl_module.device).view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean)


@paddle.no_grad()
def compute_irtr_itm_itc_recall(pl_module, split):
    print("[Evaluation] load irtr dataset for text features caching")
    paddle.cuda.empty_cache()
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_loader = paddle.utils.data.DataLoader(
        text_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_text"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=text_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )

    print("[Evaluation] load irtr dataset for image features caching")
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split, image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = paddle.utils.data.DataLoader(
        image_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_image"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=image_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )

    print("[Evaluation] start to cache the text features")
    text_embedss_cache, extend_text_masks_cache, unimodal_feats_text_cache, tiids = list(), list(), list(), list()

    if pl_module.hparams.config["amp_flag"]:
        with paddle.cuda.amp.autocast():
            for _b in tqdm(text_loader, desc="text prefetch loop"):
                text_embedss, extend_text_masks, unimodal_feats_text = pl_module.infer_text(
                    {
                        "text_ids": _b["text_ids"].to(pl_module.device),
                        "text_masks": _b["text_masks"].to(pl_module.device),
                        "text_labels": _b["text_labels"].to(pl_module.device),
                    },
                    itc=True,
                )
                text_embedss_cache.append(text_embedss)
                unimodal_feats_text_cache.append(unimodal_feats_text)
                extend_text_masks_cache.append(extend_text_masks)
                tiids += _b["img_index"]
    else:
        for _b in tqdm(text_loader, desc="text prefetch loop"):
            text_embedss, extend_text_masks, unimodal_feats_text = pl_module.infer_text(
                {
                    "text_ids": _b["text_ids"].to(pl_module.device),
                    "text_masks": _b["text_masks"].to(pl_module.device),
                    "text_labels": _b["text_labels"].to(pl_module.device),
                },
                itc=True,
            )
            text_embedss_cache.append(text_embedss)
            unimodal_feats_text_cache.append(unimodal_feats_text)
            extend_text_masks_cache.append(extend_text_masks)
            tiids += _b["img_index"]

    text_embedss_cache = paddle.cat(text_embedss_cache, dim=1)
    unimodal_feats_text_cache = paddle.cat(unimodal_feats_text_cache, dim=0)
    extend_text_masks_cache = paddle.cat(extend_text_masks_cache, dim=0)
    tiids = paddle.LongTensor(tiids)

    print("[Evaluation] gather all texts")
    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()

    text_embedss_cache = (
        pl_module.all_gather(text_embedss_cache.transpose(0, 1))
        .to(pl_module.device)
        .view(-1, text_embedss_cache.size(0), text_embedss_cache.size(2), text_embedss_cache.size(3))
        .transpose(0, 1)
    )
    unimodal_feats_text_cache = (
        pl_module.all_gather(unimodal_feats_text_cache)
        .view(-1, unimodal_feats_text_cache.size(1))
        .to(pl_module.device)
    )
    extend_text_masks_cache = (
        pl_module.all_gather(extend_text_masks_cache)
        .to(pl_module.device)
        .view(-1, extend_text_masks_cache.size(1), extend_text_masks_cache.size(2), extend_text_masks_cache.size(3))
    )
    tiids = pl_module.all_gather(tiids).to(pl_module.device).view(-1)

    print("[Evaluation] start to cache the image features")
    image_embedss_cache, unimodal_feats_image_cache, iids_cache = list(), list(), list()
    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = list()
    if pl_module.hparams.config["amp_flag"]:
        with paddle.cuda.amp.autocast():
            for _b in tqdm(image_loader, desc="image prefetch loop"):
                img_input = _b["image"][0].to(pl_module.device)
                image_embedss, unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)
                image_embedss_cache.append(image_embedss)
                if pl_module.hparams.config["gather_all_image_inputs"]:
                    img_input_cache.append(img_input)
                unimodal_feats_image_cache.append(unimodal_feats_image)
                iids_cache += _b["img_index"]
    else:
        for _b in tqdm(image_loader, desc="image prefetch loop"):
            img_input = _b["image"][0].to(pl_module.device)
            image_embedss, unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)
            image_embedss_cache.append(image_embedss)
            if pl_module.hparams.config["gather_all_image_inputs"]:
                img_input_cache.append(img_input)
            unimodal_feats_image_cache.append(unimodal_feats_image)
            iids_cache += _b["img_index"]
    image_embedss_cache = paddle.cat(image_embedss_cache, dim=1)
    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = paddle.cat(img_input_cache, dim=0)
    unimodal_feats_image_cache = paddle.cat(unimodal_feats_image_cache, dim=0)

    # top-k contrastive scores
    print("[Evaluation] start to compute the irtr recall")

    print("[Evaluation] start image-to-text")

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    _, topk_idx = sims_matrix.topk(k=pl_module.hparams.config["k_test"], dim=1)

    paddle.cuda.empty_cache()

    image_index, rank_scores, rank_iids = 0, list(), list()
    for _iid in tqdm(iids_cache, desc="image-to-text rank loop"):
        topk_idx_i = topk_idx[image_index]
        image_embedss = image_embedss_cache[:, image_index]
        text_embedss = text_embedss_cache[:, topk_idx_i]
        extend_text_masks = extend_text_masks_cache[topk_idx_i]
        if pl_module.hparams.config["image_chunks"] >= 2:
            text_embedss = paddle.chunk(text_embedss, pl_module.hparams.config["text_chunks"], dim=1)
            extend_text_masks = paddle.chunk(extend_text_masks, pl_module.hparams.config["text_chunks"], dim=0)
            score_list, img_batch_score = [], None
            for text_embedss_, extend_text_masks_ in zip(text_embedss, extend_text_masks):
                if pl_module.hparams.config["amp_flag"]:
                    with paddle.cuda.amp.autocast():
                        score = pl_module.itm_score(
                            pl_module.infer_fusion(
                                image_embedss,
                                text_embedss_,
                                extend_text_masks_,
                                irtr_len_image=text_embedss_.size(1),
                            )["cls_feats"]
                        )[:, 1]
                        if img_batch_score is None:
                            img_batch_score = paddle.full(
                                (sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device
                            )
                        score_list.append(score)

                else:
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embedss,
                            text_embedss_,
                            extend_text_masks_,
                            irtr_len_image=text_embedss_.size(1),
                        )["cls_feats"]
                    )[:, 1]
                    if img_batch_score is None:
                        img_batch_score = paddle.full(
                            (sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device
                        )
                    score_list.append(score)
                img_batch_score[topk_idx_i] = paddle.cat(score_list, dim=0)
        else:
            if pl_module.hparams.config["amp_flag"]:
                with paddle.cuda.amp.autocast():
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embedss,
                            text_embedss,
                            extend_text_masks,
                            irtr_len_image=text_embedss.size(1),
                        )["cls_feats"]
                    )[:, 1]
                    img_batch_score = paddle.full(
                        (sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device
                    )
                    img_batch_score[topk_idx_i] = score
            else:
                score = pl_module.itm_score(
                    pl_module.infer_fusion(
                        image_embedss,
                        text_embedss,
                        extend_text_masks,
                        irtr_len_image=text_embedss.size(1),
                    )["cls_feats"]
                )[:, 1]
                img_batch_score = paddle.full(
                    (sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device
                )
                img_batch_score[topk_idx_i] = score
        rank_scores.append(img_batch_score)
        rank_iids.append(_iid)

        image_index += 1
    rank_iids = paddle.LongTensor(rank_iids)
    rank_scores = paddle.cat(rank_scores, dim=0)
    print("[Evaluation] start text-to-image")

    unimodal_feats_image_cache = (
        pl_module.all_gather(unimodal_feats_image_cache)
        .to(pl_module.device)
        .view(-1, unimodal_feats_image_cache.size(1))
    )

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    _, topk_idx = sims_matrix.topk(k=pl_module.hparams.config["k_test"], dim=0)
    rank = paddle.distributed.get_rank()
    del unimodal_feats_image_cache, unimodal_feats_text_cache
    import gc

    gc.collect()
    paddle.cuda.empty_cache()
    print("[Evaluation] gather all images")
    # if out of memory, then let's gather all the image input and rerun the vision part, but slower 4~5 times

    if text_embedss_cache.size(1) % paddle.distributed.get_world_size() == 0:
        step = text_embedss_cache.size(1) // paddle.distributed.get_world_size()
    else:
        step = text_embedss_cache.size(1) // paddle.distributed.get_world_size() + 1
    start = rank * step
    end = min(text_embedss_cache.size(1), (rank + 1) * step)
    text_embedss_cache = text_embedss_cache[:, start:end]
    extend_text_masks_cache = extend_text_masks_cache[start:end]
    # topk_idx = topk_idx[:, start:end]

    if pl_module.hparams.config["gather_all_image_inputs"]:
        if not pl_module.hparams.config["save_memory"]:
            img_input_cache = (
                pl_module.all_gather(img_input_cache)
                .to(pl_module.device)
                .view(-1, img_input_cache.size(1), img_input_cache.size(2), img_input_cache.size(3))
            )
        else:
            useful_num = topk_idx.tolist()
            print(len(useful_num), len(useful_num[0]))
            useful_num = [item for sublist in useful_num for item in sublist]
            useful_num = set(useful_num)
            print(len(useful_num))
            all_idx_matrix = paddle.zeros(sims_matrix.size(0)).long().to(pl_module.device)
            for i in range(topk_idx.size(1)):
                all_idx_matrix[topk_idx[:, i]] = 1

            image_input_list, image_input_idx_list = [], []
            current_image_num = sims_matrix.size(0) // dist.get_world_size()
            for i in range(current_image_num):
                j = i + current_image_num * rank
                if all_idx_matrix[j] == 1:
                    image_input_list.append(img_input_cache[i])
                    image_input_idx_list.append(j)
            image_input_list = paddle.stack(image_input_list, dim=0)
            image_input_idx_list = paddle.LongTensor(image_input_idx_list)
            img_input_cache = image_input_list

            gather_img_input_cache = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_img_input_cache, img_input_cache)
            gather_img_input_cache = [i.to(pl_module.device) for i in gather_img_input_cache]
            gather_img_input_cache = paddle.cat(gather_img_input_cache, dim=0)
            img_input_cache = gather_img_input_cache

            gather_image_input_idx_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_image_input_idx_list, image_input_idx_list)
            gather_image_input_idx_list = [i.to(pl_module.device) for i in gather_image_input_idx_list]
            gather_image_input_idx_list = paddle.cat(gather_image_input_idx_list, dim=0)
            image_input_idx_list = gather_image_input_idx_list

            print(img_input_cache.shape, image_input_idx_list.shape)

            inverse_img_input_idx = paddle.zeros(sims_matrix.size(0)).long().fill_(-1).to(pl_module.device)
            for i in range(image_input_idx_list.size(0)):
                inverse_img_input_idx[image_input_idx_list[i]] = i

    else:
        if not pl_module.hparams.config["save_memory"]:
            image_embedss_cache = (
                pl_module.all_gather(image_embedss_cache.transpose(0, 1))
                .to(pl_module.device)
                .view(-1, image_embedss_cache.size(0), image_embedss_cache.size(2), image_embedss_cache.size(3))
                .transpose(0, 1)
            )
        else:
            useful_num = topk_idx.tolist()
            print(len(useful_num), len(useful_num[0]))
            useful_num = [item for sublist in useful_num for item in sublist]
            useful_num = set(useful_num)
            print(len(useful_num))

            all_idx_matrix = paddle.zeros(sims_matrix.size(0)).long().to(pl_module.device)
            for i in range(topk_idx.size(1)):
                all_idx_matrix[topk_idx[:, i]] = 1
            # current_idx_matrix = paddle.zeros(sims_matrix.size(0))
            # for i in range(end-start):
            #     current_idx_matrix[topk_idx[:, i]] = 1
            image_embedss_cache = image_embedss_cache.transpose(0, 1)
            image_embedss_list, image_embedss_idx_list = [], []
            current_image_num = sims_matrix.size(0) // dist.get_world_size()
            for i in range(current_image_num):
                j = i + current_image_num * rank
                if all_idx_matrix[j] == 1:
                    image_embedss_list.append(image_embedss_cache[i])
                    image_embedss_idx_list.append(j)
            image_embedss_list = paddle.stack(image_embedss_list, dim=0)
            image_embedss_idx_list = paddle.LongTensor(image_embedss_idx_list)
            image_embedss_cache = image_embedss_list

            gather_image_embedss_cache = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_image_embedss_cache, image_embedss_cache)
            gather_image_embedss_cache = [i.to(pl_module.device) for i in gather_image_embedss_cache]
            gather_image_embedss_cache = paddle.cat(gather_image_embedss_cache, dim=0)
            image_embedss_cache = gather_image_embedss_cache

            gather_image_embedss_idx_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_image_embedss_idx_list, image_embedss_idx_list)
            gather_image_embedss_idx_list = [i.to(pl_module.device) for i in gather_image_embedss_idx_list]
            gather_image_embedss_idx_list = paddle.cat(gather_image_embedss_idx_list, dim=0)
            image_embedss_idx_list = gather_image_embedss_idx_list

            print(image_embedss_cache.shape, image_embedss_idx_list.shape)
            image_embedss_cache = image_embedss_cache.transpose(0, 1)

            inverse_image_embedss_idx = paddle.zeros(sims_matrix.size(0)).long().fill_(-1).to(pl_module.device)
            for i in range(image_embedss_idx_list.size(0)):
                inverse_image_embedss_idx[image_embedss_idx_list[i]] = i

    topk_idx = topk_idx[:, start:end]

    txt_rank_scores = list()
    for text_index in tqdm(range(end - start), desc="text-to-image rank loop"):
        topk_idx_i = topk_idx[:, text_index]
        if pl_module.hparams.config["gather_all_image_inputs"]:
            if pl_module.hparams.config["save_memory"]:
                img_input = img_input_cache[inverse_img_input_idx[topk_idx_i]]
            else:
                img_input = img_input_cache[topk_idx_i]
            if pl_module.hparams.config["amp_flag"]:
                with paddle.cuda.amp.autocast():
                    image_embedss = pl_module.infer_image(img=img_input)
            else:
                image_embedss = pl_module.infer_image(img=img_input)
        else:
            if pl_module.hparams.config["save_memory"]:
                image_embedss = image_embedss_cache[:, inverse_image_embedss_idx[topk_idx_i]]
            else:
                image_embedss = image_embedss_cache[:, topk_idx_i]
        text_embedss = text_embedss_cache[:, text_index]
        extend_text_masks = (
            extend_text_masks_cache[text_index]
            .unsqueeze_(0)
            .expand(
                image_embedss.size(1),
                extend_text_masks_cache.size(1),
                extend_text_masks_cache.size(2),
                extend_text_masks_cache.size(3),
            )
        )
        if pl_module.hparams.config["image_chunks"] >= 2:
            image_embedss = paddle.chunk(image_embedss, pl_module.hparams.config["image_chunks"], dim=1)
            extend_text_masks = paddle.chunk(extend_text_masks, pl_module.hparams.config["image_chunks"], dim=0)
            score_list, txt_batch_score = [], None
            for image_embedss_, extend_text_masks_ in zip(image_embedss, extend_text_masks):
                if pl_module.hparams.config["amp_flag"]:
                    with paddle.cuda.amp.autocast():
                        score = pl_module.itm_score(
                            pl_module.infer_fusion(
                                image_embedss_,
                                text_embedss,
                                extend_text_masks_,
                                irtr_len_text=image_embedss_.size(1),
                            )["cls_feats"]
                        )[:, 1]
                        if txt_batch_score is None:
                            txt_batch_score = paddle.full(
                                (sims_matrix.size(0),), -100.0, dtype=score.dtype, device=pl_module.device
                            )
                        score_list.append(score)
                else:
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embedss_,
                            text_embedss,
                            extend_text_masks_,
                            irtr_len_text=image_embedss_.size(1),
                        )["cls_feats"]
                    )[:, 1]
                    if txt_batch_score is None:
                        txt_batch_score = paddle.full(
                            (sims_matrix.size(0),), -100.0, dtype=score.dtype, device=pl_module.device
                        )
                    score_list.append(score)
            txt_batch_score[topk_idx_i] = paddle.cat(score_list, dim=0)
        else:
            if pl_module.hparams.config["amp_flag"]:
                with paddle.cuda.amp.autocast():
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embedss,
                            text_embedss,
                            extend_text_masks,
                            irtr_len_text=image_embedss.size(1),
                        )["cls_feats"]
                    )[:, 1]
                    txt_batch_score = paddle.full(
                        (sims_matrix.size(0),), -100.0, dtype=score.dtype, device=pl_module.device
                    )
                    txt_batch_score[topk_idx_i] = score
            else:
                score = pl_module.itm_score(
                    pl_module.infer_fusion(
                        image_embedss,
                        text_embedss,
                        extend_text_masks,
                        irtr_len_text=image_embedss.size(1),
                    )["cls_feats"]
                )[:, 1]
                txt_batch_score = paddle.full(
                    (sims_matrix.size(0),), -100.0, dtype=score.dtype, device=pl_module.device
                )
                txt_batch_score[topk_idx_i] = score
        txt_rank_scores.append(txt_batch_score)
    txt_rank_scores = paddle.cat(txt_rank_scores, dim=0)

    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()

    iids = pl_module.all_gather(rank_iids).to(pl_module.device).view(-1)
    img_scores = pl_module.all_gather(rank_scores).to(pl_module.device).view(len(iids), -1)
    txt_scores = pl_module.all_gather(txt_rank_scores).to(pl_module.device).view(-1, len(iids)).t()

    scores = paddle.stack((img_scores, txt_scores), dim=-1)
    scores = paddle.max(scores, dim=-1)[0]

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    del text_embedss_cache, extend_text_masks_cache, image_embedss_cache
    if pl_module.hparams.config["gather_all_image_inputs"]:
        del img_input_cache
    import gc

    gc.collect()
    paddle.cuda.empty_cache()
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean)


@paddle.no_grad()
def compute_irtr_itm_itc_recall_meter(pl_module, split):
    print("[Evaluation] load irtr dataset for text features caching")
    paddle.cuda.empty_cache()
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_loader = paddle.utils.data.DataLoader(
        text_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_text"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=text_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )

    print("[Evaluation] load irtr dataset for image features caching")
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split, image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = paddle.utils.data.DataLoader(
        image_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_image"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=image_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )

    print("[Evaluation] start to cache the text features")
    text_embeds_cache, extend_text_masks_cache, unimodal_feats_text_cache, tiids = list(), list(), list(), list()

    if pl_module.hparams.config["amp_flag"]:
        with paddle.cuda.amp.autocast():
            for _b in tqdm(text_loader, desc="text prefetch loop"):
                text_embeds, extend_text_masks, unimodal_feats_text = pl_module.infer_text(
                    {
                        "text_ids": _b["text_ids"].to(pl_module.device),
                        "text_masks": _b["text_masks"].to(pl_module.device),
                        "text_labels": _b["text_labels"].to(pl_module.device),
                    },
                    itc=True,
                )
                text_embeds_cache.append(text_embeds)
                unimodal_feats_text_cache.append(unimodal_feats_text)
                extend_text_masks_cache.append(extend_text_masks)
                tiids += _b["img_index"]
    else:
        for _b in tqdm(text_loader, desc="text prefetch loop"):
            text_embeds, extend_text_masks, unimodal_feats_text = pl_module.infer_text(
                {
                    "text_ids": _b["text_ids"].to(pl_module.device),
                    "text_masks": _b["text_masks"].to(pl_module.device),
                    "text_labels": _b["text_labels"].to(pl_module.device),
                },
                itc=True,
            )
            text_embeds_cache.append(text_embeds)
            unimodal_feats_text_cache.append(unimodal_feats_text)
            extend_text_masks_cache.append(extend_text_masks)
            tiids += _b["img_index"]

    text_embeds_cache = paddle.cat(text_embeds_cache, dim=0)
    unimodal_feats_text_cache = paddle.cat(unimodal_feats_text_cache, dim=0)
    extend_text_masks_cache = paddle.cat(extend_text_masks_cache, dim=0)
    tiids = paddle.LongTensor(tiids)

    print("[Evaluation] gather all texts")
    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()

    text_embeds_cache = (
        pl_module.all_gather(text_embeds_cache)
        .to(pl_module.device)
        .view(-1, text_embeds_cache.size(1), text_embeds_cache.size(2))
    )
    unimodal_feats_text_cache = (
        pl_module.all_gather(unimodal_feats_text_cache)
        .view(-1, unimodal_feats_text_cache.size(1))
        .to(pl_module.device)
    )
    extend_text_masks_cache = (
        pl_module.all_gather(extend_text_masks_cache)
        .to(pl_module.device)
        .view(-1, extend_text_masks_cache.size(1), extend_text_masks_cache.size(2), extend_text_masks_cache.size(3))
    )
    tiids = pl_module.all_gather(tiids).to(pl_module.device).view(-1)

    print("[Evaluation] start to cache the image features")
    image_embeds_cache, unimodal_feats_image_cache, iids_cache = list(), list(), list()
    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = list()
    if pl_module.hparams.config["amp_flag"]:
        with paddle.cuda.amp.autocast():
            for _b in tqdm(image_loader, desc="image prefetch loop"):
                img_input = _b["image"][0].to(pl_module.device)
                image_embeds, unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)
                image_embeds_cache.append(image_embeds)
                if pl_module.hparams.config["gather_all_image_inputs"]:
                    img_input_cache.append(img_input)
                unimodal_feats_image_cache.append(unimodal_feats_image)
                iids_cache += _b["img_index"]
    else:
        for _b in tqdm(image_loader, desc="image prefetch loop"):
            img_input = _b["image"][0].to(pl_module.device)
            image_embeds, unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)
            image_embeds_cache.append(image_embeds)
            if pl_module.hparams.config["gather_all_image_inputs"]:
                img_input_cache.append(img_input)
            unimodal_feats_image_cache.append(unimodal_feats_image)
            iids_cache += _b["img_index"]
    image_embeds_cache = paddle.cat(image_embeds_cache, dim=0)
    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = paddle.cat(img_input_cache, dim=0)
    unimodal_feats_image_cache = paddle.cat(unimodal_feats_image_cache, dim=0)

    # top-k contrastive scores
    print("[Evaluation] start to compute the irtr recall")

    print("[Evaluation] start image-to-text")

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    _, topk_idx = sims_matrix.topk(k=pl_module.hparams.config["k_test"], dim=1)

    paddle.cuda.empty_cache()

    image_index, rank_scores, rank_iids = 0, list(), list()
    for _iid in tqdm(iids_cache, desc="image-to-text rank loop"):
        topk_idx_i = topk_idx[image_index]
        image_embeds = image_embeds_cache[image_index]
        text_embeds = text_embeds_cache[topk_idx_i]
        extend_text_masks = extend_text_masks_cache[topk_idx_i]
        if pl_module.hparams.config["image_chunks"] >= 2:
            text_embeds = paddle.chunk(text_embeds, pl_module.hparams.config["text_chunks"], dim=0)
            extend_text_masks = paddle.chunk(extend_text_masks, pl_module.hparams.config["text_chunks"], dim=0)
            score_list, img_batch_score = [], None
            for text_embeds_, extend_text_masks_ in zip(text_embeds, extend_text_masks):
                if pl_module.hparams.config["amp_flag"]:
                    with paddle.cuda.amp.autocast():
                        score = pl_module.itm_score(
                            pl_module.infer_fusion(
                                image_embeds,
                                text_embeds_,
                                extend_text_masks_,
                                irtr_len_image=text_embeds_.size(0),
                            )["cls_feats"]
                        )[:, 1]
                        if img_batch_score is None:
                            img_batch_score = paddle.full(
                                (sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device
                            )
                        score_list.append(score)

                else:
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embeds,
                            text_embeds_,
                            extend_text_masks_,
                            irtr_len_image=text_embeds_.size(0),
                        )["cls_feats"]
                    )[:, 1]
                    if img_batch_score is None:
                        img_batch_score = paddle.full(
                            (sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device
                        )
                    score_list.append(score)
                img_batch_score[topk_idx_i] = paddle.cat(score_list, dim=0)
        else:
            if pl_module.hparams.config["amp_flag"]:
                with paddle.cuda.amp.autocast():
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embeds,
                            text_embeds,
                            extend_text_masks,
                            irtr_len_image=text_embeds.size(0),
                        )["cls_feats"]
                    )[:, 1]
                    img_batch_score = paddle.full(
                        (sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device
                    )
                    img_batch_score[topk_idx_i] = score
            else:
                score = pl_module.itm_score(
                    pl_module.infer_fusion(
                        image_embeds,
                        text_embeds,
                        extend_text_masks,
                        irtr_len_image=text_embeds.size(0),
                    )["cls_feats"]
                )[:, 1]
                img_batch_score = paddle.full(
                    (sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device
                )
                img_batch_score[topk_idx_i] = score
        rank_scores.append(img_batch_score)
        rank_iids.append(_iid)

        image_index += 1
    rank_iids = paddle.LongTensor(rank_iids)
    rank_scores = paddle.cat(rank_scores, dim=0)
    print("[Evaluation] start text-to-image")

    unimodal_feats_image_cache = (
        pl_module.all_gather(unimodal_feats_image_cache)
        .to(pl_module.device)
        .view(-1, unimodal_feats_image_cache.size(1))
    )

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    _, topk_idx = sims_matrix.topk(k=pl_module.hparams.config["k_test"], dim=0)
    rank = paddle.distributed.get_rank()
    del unimodal_feats_image_cache, unimodal_feats_text_cache
    import gc

    gc.collect()
    paddle.cuda.empty_cache()
    print("[Evaluation] gather all images")
    # if out of memory, then let's gather all the image input and rerun the vision part, but slower 4~5 times

    if text_embeds_cache.size(0) % paddle.distributed.get_world_size() == 0:
        step = text_embeds_cache.size(0) // paddle.distributed.get_world_size()
    else:
        step = text_embeds_cache.size(0) // paddle.distributed.get_world_size() + 1
    start = rank * step
    end = min(text_embeds_cache.size(0), (rank + 1) * step)
    text_embeds_cache = text_embeds_cache[start:end]
    extend_text_masks_cache = extend_text_masks_cache[start:end]
    topk_idx = topk_idx[:, start:end]

    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = (
            pl_module.all_gather(img_input_cache)
            .to(pl_module.device)
            .view(-1, img_input_cache.size(1), img_input_cache.size(2), img_input_cache.size(3))
        )
    else:
        image_embeds_cache = (
            pl_module.all_gather(image_embeds_cache)
            .to(pl_module.device)
            .view(-1, image_embeds_cache.size(1), image_embeds_cache.size(2))
        )

    txt_rank_scores = list()
    for text_index in tqdm(range(end - start), desc="text-to-image rank loop"):
        topk_idx_i = topk_idx[:, text_index]
        if pl_module.hparams.config["gather_all_image_inputs"]:
            img_input = img_input_cache[topk_idx_i]
            if pl_module.hparams.config["amp_flag"]:
                with paddle.cuda.amp.autocast():
                    image_embeds = pl_module.infer_image(img=img_input)
            else:
                image_embeds = pl_module.infer_image(img=img_input)
        else:
            image_embeds = image_embeds_cache[topk_idx_i]
        text_embeds = text_embeds_cache[text_index]
        extend_text_masks = (
            extend_text_masks_cache[text_index]
            .unsqueeze_(0)
            .expand(
                image_embeds.size(0),
                extend_text_masks_cache.size(1),
                extend_text_masks_cache.size(2),
                extend_text_masks_cache.size(3),
            )
        )
        if pl_module.hparams.config["image_chunks"] >= 2:
            image_embeds = paddle.chunk(image_embeds, pl_module.hparams.config["image_chunks"], dim=0)
            extend_text_masks = paddle.chunk(extend_text_masks, pl_module.hparams.config["image_chunks"], dim=0)
            score_list, txt_batch_score = [], None
            for image_embeds_, extend_text_masks_ in zip(image_embeds, extend_text_masks):
                if pl_module.hparams.config["amp_flag"]:
                    with paddle.cuda.amp.autocast():
                        score = pl_module.itm_score(
                            pl_module.infer_fusion(
                                image_embeds_,
                                text_embeds,
                                extend_text_masks_,
                                irtr_len_text=image_embeds_.size(0),
                            )["cls_feats"]
                        )[:, 1]
                        if txt_batch_score is None:
                            txt_batch_score = paddle.full(
                                (sims_matrix.size(0),), -100.0, dtype=score.dtype, device=pl_module.device
                            )
                        score_list.append(score)
                else:
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embeds_,
                            text_embeds,
                            extend_text_masks_,
                            irtr_len_text=image_embeds_.size(0),
                        )["cls_feats"]
                    )[:, 1]
                    if txt_batch_score is None:
                        txt_batch_score = paddle.full(
                            (sims_matrix.size(0),), -100.0, dtype=score.dtype, device=pl_module.device
                        )
                    score_list.append(score)
            txt_batch_score[topk_idx_i] = paddle.cat(score_list, dim=0)
        else:
            if pl_module.hparams.config["amp_flag"]:
                with paddle.cuda.amp.autocast():
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embeds,
                            text_embeds,
                            extend_text_masks,
                            irtr_len_text=image_embeds.size(0),
                        )["cls_feats"]
                    )[:, 1]
                    txt_batch_score = paddle.full(
                        (sims_matrix.size(0),), -100.0, dtype=score.dtype, device=pl_module.device
                    )
                    txt_batch_score[topk_idx_i] = score
            else:
                score = pl_module.itm_score(
                    pl_module.infer_fusion(
                        image_embeds,
                        text_embeds,
                        extend_text_masks,
                        irtr_len_text=image_embeds.size(0),
                    )["cls_feats"]
                )[:, 1]
                txt_batch_score = paddle.full(
                    (sims_matrix.size(0),), -100.0, dtype=score.dtype, device=pl_module.device
                )
                txt_batch_score[topk_idx_i] = score
        txt_rank_scores.append(txt_batch_score)
    txt_rank_scores = paddle.cat(txt_rank_scores, dim=0)

    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()

    iids = pl_module.all_gather(rank_iids).to(pl_module.device).view(-1)
    img_scores = pl_module.all_gather(rank_scores).to(pl_module.device).view(len(iids), -1)
    txt_scores = pl_module.all_gather(txt_rank_scores).to(pl_module.device).view(-1, len(iids)).t()

    scores = paddle.stack((img_scores, txt_scores), dim=-1)
    scores = paddle.max(scores, dim=-1)[0]

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    del text_embeds_cache, extend_text_masks_cache, image_embeds_cache
    if pl_module.hparams.config["gather_all_image_inputs"]:
        del img_input_cache
    import gc

    gc.collect()
    paddle.cuda.empty_cache()
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean)


@paddle.no_grad()
def compute_irtr_itc_recall(pl_module, split):
    print("[Evaluation] load irtr dataset for text features caching")
    paddle.cuda.empty_cache()
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_loader = paddle.utils.data.DataLoader(
        text_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_text"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=text_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )

    print("[Evaluation] load irtr dataset for image features caching")
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split, image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = paddle.utils.data.DataLoader(
        image_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_image"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=image_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )

    print("[Evaluation] start to cache the text features")
    unimodal_feats_text_cache, tiids = list(), list()

    if pl_module.hparams.config["amp_flag"]:
        with paddle.cuda.amp.autocast():
            for _b in tqdm(text_loader, desc="text prefetch loop"):
                unimodal_feats_text = pl_module.infer_text(
                    {
                        "text_ids": _b["text_ids"].to(pl_module.device),
                        "text_masks": _b["text_masks"].to(pl_module.device),
                        "text_labels": _b["text_labels"].to(pl_module.device),
                    },
                    itc=True,
                )[2]
                unimodal_feats_text_cache.append(unimodal_feats_text)
                tiids += _b["img_index"]
    else:
        for _b in tqdm(text_loader, desc="text prefetch loop"):
            unimodal_feats_text = pl_module.infer_text(
                {
                    "text_ids": _b["text_ids"].to(pl_module.device),
                    "text_masks": _b["text_masks"].to(pl_module.device),
                    "text_labels": _b["text_labels"].to(pl_module.device),
                },
                itc=True,
            )[2]
            unimodal_feats_text_cache.append(unimodal_feats_text)
            tiids += _b["img_index"]

    unimodal_feats_text_cache = paddle.cat(unimodal_feats_text_cache, dim=0)
    tiids = paddle.LongTensor(tiids)

    print("[Evaluation] gather all texts")
    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()

    unimodal_feats_text_cache = (
        pl_module.all_gather(unimodal_feats_text_cache)
        .view(-1, unimodal_feats_text_cache.size(1))
        .to(pl_module.device)
    )
    tiids = pl_module.all_gather(tiids).to(pl_module.device).view(-1)

    print("[Evaluation] start to cache the image features")
    unimodal_feats_image_cache, iids_cache = list(), list()
    if pl_module.hparams.config["amp_flag"]:
        with paddle.cuda.amp.autocast():
            for _b in tqdm(image_loader, desc="image prefetch loop"):
                img_input = _b["image"][0].to(pl_module.device)
                unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)[1]
                unimodal_feats_image_cache.append(unimodal_feats_image)
                iids_cache += _b["img_index"]
    else:
        for _b in tqdm(image_loader, desc="image prefetch loop"):
            img_input = _b["image"][0].to(pl_module.device)
            unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)[1]
            unimodal_feats_image_cache.append(unimodal_feats_image)
            iids_cache += _b["img_index"]
    unimodal_feats_image_cache = paddle.cat(unimodal_feats_image_cache, dim=0)

    paddle.cuda.empty_cache()
    print("[Evaluation] start to compute the itc recall")

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    rank_iids = paddle.LongTensor(iids_cache)

    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()

    sims_matrix = pl_module.all_gather(sims_matrix).view(-1, sims_matrix.size(1)).to(pl_module.device)
    iids = pl_module.all_gather(rank_iids).to(pl_module.device).view(-1)
    scores = sims_matrix

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    paddle.cuda.empty_cache()
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean)


# save vqa test results to json file, then you can manually upload it to the evalai server
def vqa_test_step(pl_module, batch, output):
    try:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
            if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        )
    except Exception:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["gqa_test"].id2answer
            if "gqa_test" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["gqa"].id2answer
        )
        vqa_logits = output["vqa_logits"]
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        # questions = batch["text"]
        qids = batch["qid"]
        return {"qids": qids, "preds": vqa_preds, "gqa": True}
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    # questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": False}


def vqa_test_wrapup(outs, model_name, log_dir):
    rank = paddle.distributed.get_rank()
    qids, preds = list(), list()
    gqa = False
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out["gqa"]

    rets = list()
    for qid, pred in zip(qids, preds):
        if gqa:
            rets.append({"questionId": qid, "prediction": pred})
        else:
            rets.append({"question_id": qid, "answer": pred})

    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()

    print(f"rank: {rank}, world_size: {dist.get_world_size()}, length of rets: {len(rets)}")
    gather_rets = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gather_rets, rets)
    print(f"rank: {rank}, length of gather_rets: {len(gather_rets)}")
    print(f"rank: {rank}, length of gather_rets[0]: {len(gather_rets[0])}")

    if rank == 0:
        jsons = list()
        for rets_ in gather_rets:
            jsons += rets_
        with open(f"{log_dir}/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    if paddle.distributed.is_initialized():
        paddle.distributed.barrier()

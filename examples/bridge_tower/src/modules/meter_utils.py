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


from paddle.metric import Accuracy

from ..gadgets.my_metrics import Scalar, VQAScore
from .objectives import (
    compute_irtr_itc_recall,
    compute_irtr_itm_itc_recall,
    compute_irtr_itm_itc_recall_meter,
    compute_irtr_recall,
)


def set_metrics(model):
    for split in ["train", "val", "test"]:
        for k, v in model.hparams["loss_names"].items():
            if v <= model.hparams["task_threshold"]:
                continue
            if k == "vqa":
                setattr(model, f"{split}_{k}_score", VQAScore())
                setattr(model, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2" or k == "snli" or k == "itm" or k == "mlm":
                setattr(model, f"{split}_{k}_accuracy", Accuracy())
                setattr(model, f"{split}_{k}_loss", Scalar())
            elif k == "irtr" or k == "itc":
                setattr(model, f"{split}_{k}_loss", Scalar())
            elif k == "itm_itc":
                setattr(model, f"{split}_itm_accuracy", Accuracy())
                setattr(model, f"{split}_itm_loss", Scalar())
                setattr(model, f"{split}_itc_loss", Scalar())
            elif k == "irtr_itm_itc":
                setattr(model, f"{split}_irtr_itm_accuracy", Accuracy())
                setattr(model, f"{split}_irtr_itm_loss", Scalar())
                setattr(model, f"{split}_irtr_itc_loss", Scalar())
            else:
                raise ValueError(f"Unknown loss name: {k}")

    if model.hparams["test_only"]:
        split = "test"
    else:
        split = "val"
    setattr(model, "best_metric_log", {f"{split}/the_metric": 0})

    for k, v in model.hparams["loss_names"].items():
        if v <= model.hparams["task_threshold"]:
            continue
        if k == "vqa" and split == "val":
            getattr(model, "best_metric_log").update(
                {
                    f"{split}/{k}_score": -1e9,
                    f"{split}/{k}_loss": 1e9,
                }
            )
        if k == "nlvr2" or k == "snli":
            getattr(model, "best_metric_log").update(
                {
                    f"val/{k}_accuracy": -1e9,
                    f"test/{k}_accuracy": -1e9,
                    f"val/{k}_loss": 1e9,
                    f"test/{k}_loss": 1e9,
                }
            )
        if k == "mlm" and split == "val":
            getattr(model, "best_metric_log").update(
                {
                    f"{split}/{k}_accuracy": -1e9,
                    f"{split}/{k}_loss": 1e9,
                }
            )
        if k == "itm":
            getattr(model, "best_metric_log").update(
                {
                    f"{split}/{k}_accuracy": -1e9,
                    f"{split}/{k}_loss": 1e9,
                }
            )

        if k == "itc":
            getattr(model, "best_metric_log").update(
                {
                    f"{split}/{k}_loss": 1e9,
                }
            )

        if k == "itm_itc":
            getattr(model, "best_metric_log").update(
                {
                    f"{split}/itm_accuracy": -1e9,
                    f"{split}/itm_loss": 1e9,
                    f"{split}/itc_loss": 1e9,
                }
            )

        if k == "irtr_itm_itc":
            getattr(model, "best_metric_log").update(
                {
                    f"{split}/irtr_itm_accuracy": -1e9,
                    f"{split}/irtr_itm_loss": 1e9,
                    f"{split}/irtr_itc_loss": 1e9,
                }
            )

        if k == "irtr" or k == "irtr_itm_itc":
            getattr(model, "best_metric_log").update(
                {
                    f"{split}/ir_r1": -1e9,
                    f"{split}/ir_r5": -1e9,
                    f"{split}/ir_r10": -1e9,
                    f"{split}/tr_r1": -1e9,
                    f"{split}/tr_r5": -1e9,
                    f"{split}/tr_r10": -1e9,
                }
            )

        if k == "irtr":
            getattr(model, "best_metric_log").update(
                {
                    f"{split}/{k}_loss": 1e9,
                }
            )


def epoch_wrapup(model, split):
    the_metric = 0
    metric_log = {}

    if model.hparams["get_recall_metric"] and not model.training:
        if "irtr_itm_itc" in model.hparams["group_name"]:
            if model.hparams["model_type"] == "BT":
                (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean) = compute_irtr_itm_itc_recall(
                    model, split
                )
            else:
                if model.hparams["num_layers"] == 0:
                    (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean) = compute_irtr_itc_recall(
                        model, split
                    )
                else:
                    (
                        ir_r1,
                        ir_r5,
                        ir_r10,
                        tr_r1,
                        tr_r5,
                        tr_r10,
                        ir_mean,
                        tr_mean,
                        r_mean,
                    ) = compute_irtr_itm_itc_recall_meter(model, split)
        else:
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean) = compute_irtr_recall(model, split)
        print("### Recall metrics ###")
        print(
            f"{ir_r1 * 100:.2f} \t {ir_r5 * 100:.2f} \t {ir_r10 * 100:.2f} \t {tr_r1 * 100:.2f} \t {tr_r5 * 100:.2f} \t {tr_r10 * 100:.2f} \t {ir_mean * 100:.2f} \t {tr_mean * 100:.2f} \t {r_mean * 100:.2f}"
        )
        print("######################")
        metric_log.update(
            {
                f"{split}/ir_r1": ir_r1 * 100,
                f"{split}/ir_r5": ir_r5 * 100,
                f"{split}/ir_r10": ir_r10 * 100,
                f"{split}/tr_r1": tr_r1 * 100,
                f"{split}/tr_r5": tr_r5 * 100,
                f"{split}/tr_r10": tr_r10 * 100,
                f"{split}/ir_mean": ir_mean * 100,
                f"{split}/tr_mean": tr_mean * 100,
                f"{split}/r_mean": r_mean * 100,
            }
        )
        model.log_dict(metric_log)
        the_metric += r_mean.item()

    for loss_name, v in model.hparams["loss_names"].items():
        if v <= model.hparams["task_threshold"]:
            continue

        value = 0

        if loss_name == "vqa" and split != "test":
            value = getattr(model, f"{split}_{loss_name}_score").compute()
            model.log(f"{split}/{loss_name}/score_epoch", value)
            getattr(model, f"{split}_{loss_name}_score").reset()
            loss_epoch = getattr(model, f"{split}_{loss_name}_loss").compute()
            model.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
            getattr(model, f"{split}_{loss_name}_loss").reset()
            if split == "val":
                metric_log[f"{split}/{loss_name}_score"] = value
                metric_log[f"{split}/{loss_name}_loss"] = loss_epoch
        elif loss_name == "nlvr2" or loss_name == "snli":
            if split == "train":
                # value = getattr(model, f"{split}_{loss_name}_accuracy").compute()
                # model.log(f"{split}/{loss_name}/accuracy_epoch", value)
                # getattr(model, f"{split}_{loss_name}_accuracy").reset()
                # loss_epoch = getattr(model, f"{split}_{loss_name}_loss").compute()
                # model.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
                # getattr(model, f"{split}_{loss_name}_loss").reset()
                value = getattr(model, f"{split}_{loss_name}_accuracy").accumulate()
                # model.log(f"{split}/{loss_name}/accuracy_epoch", value)
                getattr(model, f"{split}_{loss_name}_accuracy").reset()
                loss_epoch = getattr(model, f"{split}_{loss_name}_loss").accumulate()
                # model.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
                getattr(model, f"{split}_{loss_name}_loss").reset()
                metric_dict = {
                    f"{split}/{loss_name}/accuracy_epoch": value,
                    f"{split}/{loss_name}/loss_epoch": loss_epoch,
                }
            else:
                # value1 = getattr(model, f"test_{loss_name}_accuracy").compute()
                # model.log(f"test/{loss_name}/accuracy_epoch", value1)
                # getattr(model, f"test_{loss_name}_accuracy").reset()
                # test_loss_epoch = getattr(model, f"test_{loss_name}_loss").compute()
                # model.log(f"test/{loss_name}/loss_epoch", test_loss_epoch)
                # getattr(model, f"test_{loss_name}_loss").reset()
                # metric_log[f'test/{loss_name}_accuracy'] = value1
                # metric_log[f'test/{loss_name}_loss'] = test_loss_epoch

                value1 = getattr(model, f"test_{loss_name}_accuracy").accumulate()
                # model.log(f"test/{loss_name}/accuracy_epoch", value1)
                getattr(model, f"test_{loss_name}_accuracy").reset()
                test_loss_epoch = getattr(model, f"test_{loss_name}_loss").accumulate()
                # model.log(f"test/{loss_name}/loss_epoch", test_loss_epoch)
                getattr(model, f"test_{loss_name}_loss").reset()
                metric_log[f"test/{loss_name}_accuracy"] = value1
                metric_log[f"test/{loss_name}_loss"] = test_loss_epoch

                metric_dict = {
                    f"test/{loss_name}/accuracy_epoch": value1,
                    f"test/{loss_name}/loss_epoch": test_loss_epoch,
                }

                # value = getattr(model, f"val_{loss_name}_accuracy").compute()
                # model.log(f"val/{loss_name}/accuracy_epoch", value)
                # getattr(model, f"val_{loss_name}_accuracy").reset()
                # val_loss_epoch = getattr(model, f"val_{loss_name}_loss").compute()
                # model.log(f"val/{loss_name}/loss_epoch", val_loss_epoch)
                # getattr(model, f"val_{loss_name}_loss").reset()
                # metric_log[f'val/{loss_name}_accuracy'] = value
                # metric_log[f'val/{loss_name}_loss'] = val_loss_epoch

                value = getattr(model, f"val_{loss_name}_accuracy").accumulate()
                # model.log(f"val/{loss_name}/accuracy_epoch", value)
                getattr(model, f"val_{loss_name}_accuracy").reset()
                val_loss_epoch = getattr(model, f"val_{loss_name}_loss").accumulate()
                # model.log(f"val/{loss_name}/loss_epoch", val_loss_epoch)
                getattr(model, f"val_{loss_name}_loss").reset()
                metric_log[f"val/{loss_name}_accuracy"] = value
                metric_log[f"val/{loss_name}_loss"] = val_loss_epoch

                metric_dict.update(
                    {f"val/{loss_name}/accuracy_epoch": value, f"val/{loss_name}/loss_epoch": val_loss_epoch}
                )

        elif loss_name == "itm" or loss_name == "mlm":
            # value = getattr(model, f"{split}_{loss_name}_accuracy").compute()
            value = getattr(model, f"{split}_{loss_name}_accuracy").accumulate()
            # model.log(f"{split}/{loss_name}/accuracy_epoch", value)
            getattr(model, f"{split}_{loss_name}_accuracy").reset()
            # loss_epoch = getattr(model, f"{split}_{loss_name}_loss").compute()
            loss_epoch = getattr(model, f"{split}_{loss_name}_loss").accumulate()
            # model.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
            getattr(model, f"{split}_{loss_name}_loss").reset()

            metric_dict = {f"{split}/{loss_name}/accuracy_epoch": value, f"{split}/{loss_name}/loss_epoch": loss_epoch}

            if split == "val" or (split == "test" and loss_name == "itm"):
                metric_log[f"{split}/{loss_name}_accuracy"] = value
                metric_log[f"{split}/{loss_name}_loss"] = loss_epoch
        elif loss_name == "itc":
            loss_epoch = getattr(model, f"{split}_{loss_name}_loss").compute()
            model.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
            getattr(model, f"{split}_{loss_name}_loss").reset()
            if split == "val":
                metric_log[f"{split}/{loss_name}_loss"] = loss_epoch
        elif loss_name == "itm_itc":
            value = getattr(model, f"{split}_itm_accuracy").compute()
            model.log(f"{split}/itm/accuracy_epoch", value)
            getattr(model, f"{split}_itm_accuracy").reset()
            loss_epoch = getattr(model, f"{split}_itm_loss").compute()
            model.log(f"{split}/itm/loss_epoch", loss_epoch)
            getattr(model, f"{split}_itm_loss").reset()
            loss_epoch = getattr(model, f"{split}_itc_loss").compute()
            model.log(f"{split}/itc/loss_epoch", loss_epoch)
            getattr(model, f"{split}_itc_loss").reset()
            if split == "val" or (split == "test" and loss_name == "itm"):
                metric_log[f"{split}/itm_accuracy"] = value
                metric_log[f"{split}/itm_loss"] = loss_epoch
                metric_log[f"{split}/itc_loss"] = loss_epoch
        elif loss_name == "irtr_itm_itc":
            value = getattr(model, f"{split}_irtr_itm_accuracy").compute()
            model.log(f"{split}/irtr_itm/accuracy_epoch", value)
            getattr(model, f"{split}_irtr_itm_accuracy").reset()
            loss_epoch = getattr(model, f"{split}_irtr_itm_loss").compute()
            model.log(f"{split}/irtr_itm/loss_epoch", loss_epoch)
            getattr(model, f"{split}_irtr_itm_loss").reset()
            loss_epoch = getattr(model, f"{split}_irtr_itc_loss").compute()
            model.log(f"{split}/irtr_itc/loss_epoch", loss_epoch)
            getattr(model, f"{split}_irtr_itc_loss").reset()
            if split == "val" or (split == "test" and loss_name == "itm"):
                metric_log[f"{split}/irtr_itm_accuracy"] = value
                metric_log[f"{split}/irtr_itm_loss"] = loss_epoch
                metric_log[f"{split}/irtr_itc_loss"] = loss_epoch
            value = 0
        elif loss_name == "irtr":
            loss_epoch = getattr(model, f"{split}_{loss_name}_loss").compute()
            model.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
            getattr(model, f"{split}_{loss_name}_loss").reset()
            if split != "train":
                metric_log[f"{split}/{loss_name}_loss"] = loss_epoch
            irtr_loss_epoch = loss_epoch

        the_metric += value

    # use irtr_loss for selecting checkpoints
    if model.hparams["loss_names"]["irtr"] == 1:
        the_metric = -irtr_loss_epoch

    # model.log(f"{split}/the_metric", the_metric)

    if split == "val":
        prev_the_metric = getattr(model, "best_metric_log")["val/the_metric"]
        if the_metric > prev_the_metric:
            metric_log["val/the_metric"] = the_metric
            setattr(model, "best_metric_log", metric_log)
        # model.log(f"val/best_the_metric", max(the_metric, prev_the_metric))
        metric_dict.update({"val/best_the_metric": max(the_metric, prev_the_metric)})
    if split == "test" and model.hparams["group_name"] != "vqa":
        metric_log["test/the_metric"] = the_metric
        setattr(model, "best_metric_log", metric_log)

    return metric_dict


def check_non_acc_grad(model):
    if model.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = model.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(model):
    model.current_tasks = [k for k, v in model.hparams["loss_names"].items() if v > model.hparams["task_threshold"]]
    return


def set_schedule(model, _config):
    lr = _config["learning_rate"]
    wd = _config["weight_decay"]

    no_decay = [
        "bias",
        "norm",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = [
        "vqa_classifier",
        "nlvr2_classifier",
        "snli_classifier",
        "mlm_score",
        "itm_score",
        "itc_text_head",
        "itc_image_head",
    ]
    cross_modal_names = ["cross_modal"]
    lr_mult_head = _config["lr_mult_head"]
    lr_head = lr * lr_mult_head if type(lr_mult_head) == int else lr_mult_head
    lr_mult_cross_modal = _config["lr_mult_cross_modal"]
    lr_cross_modal = lr * lr_mult_cross_modal if type(lr_mult_cross_modal) == int else lr_mult_cross_modal
    print(f"lr_head: {lr_head}, lr_cross_modal: {lr_cross_modal}")
    # end_lr = _config["end_lr"]
    # decay_power = _config["decay_power"]
    # optim_type = _config["optim_type"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "learning_rate": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "learning_rate": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "learning_rate": lr_head,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "learning_rate": lr_head,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "learning_rate": lr_cross_modal,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "learning_rate": lr_cross_modal,
        },
    ]

    return optimizer_grouped_parameters

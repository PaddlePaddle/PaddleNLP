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

from ..gadgets.my_metrics import Accuracy, Scalar, VQAScore
from .objectives import (
    compute_irtr_itc_recall,
    compute_irtr_itm_itc_recall,
    compute_irtr_itm_itc_recall_meter,
    compute_irtr_recall,
)


def set_metrics(pl_module):
    for split in ["train", "val", "test"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v <= pl_module.hparams.config["task_threshold"]:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_{k}_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2" or k == "snli" or k == "itm" or k == "mlm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "irtr" or k == "itc":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm_itc":
                setattr(pl_module, f"{split}_itm_accuracy", Accuracy())
                setattr(pl_module, f"{split}_itm_loss", Scalar())
                setattr(pl_module, f"{split}_itc_loss", Scalar())
            elif k == "irtr_itm_itc":
                setattr(pl_module, f"{split}_irtr_itm_accuracy", Accuracy())
                setattr(pl_module, f"{split}_irtr_itm_loss", Scalar())
                setattr(pl_module, f"{split}_irtr_itc_loss", Scalar())
            else:
                raise ValueError(f"Unknown loss name: {k}")

    if pl_module.hparams.config["test_only"]:
        split = "test"
    else:
        split = "val"
    setattr(pl_module, "best_metric_log", {f"{split}/the_metric": 0})

    for k, v in pl_module.hparams.config["loss_names"].items():
        if v <= pl_module.hparams.config["task_threshold"]:
            continue
        if k == "vqa" and split == "val":
            getattr(pl_module, "best_metric_log").update(
                {
                    f"{split}/{k}_score": -1e9,
                    f"{split}/{k}_loss": 1e9,
                }
            )
        if k == "nlvr2" or k == "snli":
            getattr(pl_module, "best_metric_log").update(
                {
                    f"val/{k}_accuracy": -1e9,
                    f"test/{k}_accuracy": -1e9,
                    f"val/{k}_loss": 1e9,
                    f"test/{k}_loss": 1e9,
                }
            )
        if k == "mlm" and split == "val":
            getattr(pl_module, "best_metric_log").update(
                {
                    f"{split}/{k}_accuracy": -1e9,
                    f"{split}/{k}_loss": 1e9,
                }
            )
        if k == "itm":
            getattr(pl_module, "best_metric_log").update(
                {
                    f"{split}/{k}_accuracy": -1e9,
                    f"{split}/{k}_loss": 1e9,
                }
            )

        if k == "itc":
            getattr(pl_module, "best_metric_log").update(
                {
                    f"{split}/{k}_loss": 1e9,
                }
            )

        if k == "itm_itc":
            getattr(pl_module, "best_metric_log").update(
                {
                    f"{split}/itm_accuracy": -1e9,
                    f"{split}/itm_loss": 1e9,
                    f"{split}/itc_loss": 1e9,
                }
            )

        if k == "irtr_itm_itc":
            getattr(pl_module, "best_metric_log").update(
                {
                    f"{split}/irtr_itm_accuracy": -1e9,
                    f"{split}/irtr_itm_loss": 1e9,
                    f"{split}/irtr_itc_loss": 1e9,
                }
            )

        if k == "irtr" or k == "irtr_itm_itc":
            getattr(pl_module, "best_metric_log").update(
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
            getattr(pl_module, "best_metric_log").update(
                {
                    f"{split}/{k}_loss": 1e9,
                }
            )


def epoch_wrapup(pl_module, split):
    the_metric = 0
    metric_log = {}

    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        if "irtr_itm_itc" in pl_module.hparams.config["group_name"]:
            if pl_module.hparams.config["model_type"] == "BT":
                (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean) = compute_irtr_itm_itc_recall(
                    pl_module, split
                )
            else:
                if pl_module.hparams.config["num_layers"] == 0:
                    (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean) = compute_irtr_itc_recall(
                        pl_module, split
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
                    ) = compute_irtr_itm_itc_recall_meter(pl_module, split)
        else:
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean) = compute_irtr_recall(
                pl_module, split
            )
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
        pl_module.log_dict(metric_log)
        the_metric += r_mean.item()

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v <= pl_module.hparams.config["task_threshold"]:
            continue

        value = 0

        if loss_name == "vqa" and split != "test":
            value = getattr(pl_module, f"{split}_{loss_name}_score").compute()
            pl_module.log(f"{split}/{loss_name}/score_epoch", value)
            getattr(pl_module, f"{split}_{loss_name}_score").reset()
            loss_epoch = getattr(pl_module, f"{split}_{loss_name}_loss").compute()
            pl_module.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
            getattr(pl_module, f"{split}_{loss_name}_loss").reset()
            if split == "val":
                metric_log[f"{split}/{loss_name}_score"] = value
                metric_log[f"{split}/{loss_name}_loss"] = loss_epoch
        elif loss_name == "nlvr2" or loss_name == "snli":
            if split == "train":
                value = getattr(pl_module, f"{split}_{loss_name}_accuracy").compute()
                pl_module.log(f"{split}/{loss_name}/accuracy_epoch", value)
                getattr(pl_module, f"{split}_{loss_name}_accuracy").reset()
                loss_epoch = getattr(pl_module, f"{split}_{loss_name}_loss").compute()
                pl_module.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
                getattr(pl_module, f"{split}_{loss_name}_loss").reset()
            else:
                value1 = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"test/{loss_name}/accuracy_epoch", value1)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                test_loss_epoch = getattr(pl_module, f"test_{loss_name}_loss").compute()
                pl_module.log(f"test/{loss_name}/loss_epoch", test_loss_epoch)
                getattr(pl_module, f"test_{loss_name}_loss").reset()
                metric_log[f"test/{loss_name}_accuracy"] = value1
                metric_log[f"test/{loss_name}_loss"] = test_loss_epoch

                value = getattr(pl_module, f"val_{loss_name}_accuracy").compute()
                pl_module.log(f"val/{loss_name}/accuracy_epoch", value)
                getattr(pl_module, f"val_{loss_name}_accuracy").reset()
                val_loss_epoch = getattr(pl_module, f"val_{loss_name}_loss").compute()
                pl_module.log(f"val/{loss_name}/loss_epoch", val_loss_epoch)
                getattr(pl_module, f"val_{loss_name}_loss").reset()
                metric_log[f"val/{loss_name}_accuracy"] = value
                metric_log[f"val/{loss_name}_loss"] = val_loss_epoch

        elif loss_name == "itm" or loss_name == "mlm":
            value = getattr(pl_module, f"{split}_{loss_name}_accuracy").compute()
            pl_module.log(f"{split}/{loss_name}/accuracy_epoch", value)
            getattr(pl_module, f"{split}_{loss_name}_accuracy").reset()
            loss_epoch = getattr(pl_module, f"{split}_{loss_name}_loss").compute()
            pl_module.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
            getattr(pl_module, f"{split}_{loss_name}_loss").reset()
            if split == "val" or (split == "test" and loss_name == "itm"):
                metric_log[f"{split}/{loss_name}_accuracy"] = value
                metric_log[f"{split}/{loss_name}_loss"] = loss_epoch
        elif loss_name == "itc":
            loss_epoch = getattr(pl_module, f"{split}_{loss_name}_loss").compute()
            pl_module.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
            getattr(pl_module, f"{split}_{loss_name}_loss").reset()
            if split == "val":
                metric_log[f"{split}/{loss_name}_loss"] = loss_epoch
        elif loss_name == "itm_itc":
            value = getattr(pl_module, f"{split}_itm_accuracy").compute()
            pl_module.log(f"{split}/itm/accuracy_epoch", value)
            getattr(pl_module, f"{split}_itm_accuracy").reset()
            loss_epoch = getattr(pl_module, f"{split}_itm_loss").compute()
            pl_module.log(f"{split}/itm/loss_epoch", loss_epoch)
            getattr(pl_module, f"{split}_itm_loss").reset()
            loss_epoch = getattr(pl_module, f"{split}_itc_loss").compute()
            pl_module.log(f"{split}/itc/loss_epoch", loss_epoch)
            getattr(pl_module, f"{split}_itc_loss").reset()
            if split == "val" or (split == "test" and loss_name == "itm"):
                metric_log[f"{split}/itm_accuracy"] = value
                metric_log[f"{split}/itm_loss"] = loss_epoch
                metric_log[f"{split}/itc_loss"] = loss_epoch
        elif loss_name == "irtr_itm_itc":
            value = getattr(pl_module, f"{split}_irtr_itm_accuracy").compute()
            pl_module.log(f"{split}/irtr_itm/accuracy_epoch", value)
            getattr(pl_module, f"{split}_irtr_itm_accuracy").reset()
            loss_epoch = getattr(pl_module, f"{split}_irtr_itm_loss").compute()
            pl_module.log(f"{split}/irtr_itm/loss_epoch", loss_epoch)
            getattr(pl_module, f"{split}_irtr_itm_loss").reset()
            loss_epoch = getattr(pl_module, f"{split}_irtr_itc_loss").compute()
            pl_module.log(f"{split}/irtr_itc/loss_epoch", loss_epoch)
            getattr(pl_module, f"{split}_irtr_itc_loss").reset()
            if split == "val" or (split == "test" and loss_name == "itm"):
                metric_log[f"{split}/irtr_itm_accuracy"] = value
                metric_log[f"{split}/irtr_itm_loss"] = loss_epoch
                metric_log[f"{split}/irtr_itc_loss"] = loss_epoch
            value = 0
        elif loss_name == "irtr":
            loss_epoch = getattr(pl_module, f"{split}_{loss_name}_loss").compute()
            pl_module.log(f"{split}/{loss_name}/loss_epoch", loss_epoch)
            getattr(pl_module, f"{split}_{loss_name}_loss").reset()
            if split != "train":
                metric_log[f"{split}/{loss_name}_loss"] = loss_epoch
            irtr_loss_epoch = loss_epoch

        the_metric += value

    # use irtr_loss for selecting checkpoints
    if pl_module.hparams.config["loss_names"]["irtr"] == 1:
        the_metric = -irtr_loss_epoch

    pl_module.log(f"{split}/the_metric", the_metric)

    if split == "val":
        prev_the_metric = getattr(pl_module, "best_metric_log")["val/the_metric"]
        if the_metric > prev_the_metric:
            metric_log["val/the_metric"] = the_metric
            setattr(pl_module, "best_metric_log", metric_log)
        pl_module.log("val/best_the_metric", max(the_metric, prev_the_metric))
    if split == "test" and pl_module.hparams.config["group_name"] != "vqa":
        metric_log["test/the_metric"] = the_metric
        setattr(pl_module, "best_metric_log", metric_log)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v > pl_module.hparams.config["task_threshold"]
    ]
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
            "lr": lr,
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
            "lr": lr,
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
            "lr": lr_head,
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
            "lr": lr_head,
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
            "lr": lr_cross_modal,
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
            "lr": lr_cross_modal,
        },
    ]
    # breakpoint()

    # if optim_type == "adamw":
    #     optimizer = AdamW(
    #         parameters=optimizer_grouped_parameters, learning_rate=lr, epsilon=1e-8, beta1=0.9, beta2=0.98
    #     )
    # elif optim_type == "adam":
    #     optimizer = torch.optim.Adam(optimizer_grouped_parameters, learning_rate=lr)
    # elif optim_type == "sgd":
    #     optimizer = torch.optim.SGD(optimizer_grouped_parameters, learning_rate=lr, momentum=0.9)

    # if _config["max_steps"] == -1:
    #     max_steps = (
    #         len(model.trainer.datamodule.train_dataloader())
    #         * _config["max_epochs"]
    #         // model.trainer.accumulate_grad_batches
    #         # // pl_module.trainer.gpus
    #         # Very Strange that the len(pl_module.trainer.datamodule.train_dataloader()) haven't been divided by pl_module.trainer.gpus, So I fix it to get the correct max_steps for scheduler.
    #         # Alert: Only when the replace_sampler_ddp is True, the len(pl_module.trainer.datamodule.train_dataloader()) need to be divided by pl_module.trainer.gpus.
    #     )
    # else:
    #     max_steps = _config["max_steps"]

    # warmup_steps = _config["warmup_steps"]
    # if isinstance(_config["warmup_steps"], float):
    #     warmup_steps = int(max_steps * warmup_steps)

    # if decay_power == "cosine":
    #     scheduler = get_cosine_schedule_with_warmup(
    #         optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    #     )
    # else:
    #     scheduler = get_polynomial_decay_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=warmup_steps,
    #         num_training_steps=max_steps,
    #         lr_end=end_lr,
    #         power=decay_power,
    #     )

    # sched = {"scheduler": scheduler, "interval": "step"}

    # return (
    #     [optimizer],
    #     [sched],
    # )
    return optimizer_grouped_parameters

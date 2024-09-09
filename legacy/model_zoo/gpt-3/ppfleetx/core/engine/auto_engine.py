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

import os
import sys
import numpy as np

import paddle
import paddle.base.core as core
import paddle.nn as nn
from paddle.distributed.fleet import auto
from paddle.profiler import SummaryView
from paddle.profiler.utils import job_schedule_profiler_range

try:
    from ppfleetx.optims import build_lr_scheduler, build_optimizer
except Exception:
    pass
from ppfleetx.core.engine import BasicEngine
from ppfleetx.core.module import BasicModule
from ppfleetx.utils.device import synchronize as device_synchronize
from ppfleetx.utils.log import convert_timestamp_to_data, get_timestamp, logger
from ppfleetx.utils.version import version_check

def use_new_executor():
    new_executor_micro_batching = os.environ.get(
        'FLAGS_new_executor_micro_batching', None
    )
    return new_executor_micro_batching in [
        None,
        1,
        '1',
        True,
        'True',
        'true',
    ]

class AutoEngine(BasicEngine):
    def __init__(self, configs, module=None, mode="train"):
        super().__init__()
        version_check()

        model = None
        loss_fn = None

        if module and not isinstance(module, BasicModule):
            raise TypeError("'module' must be sub classes of `BasicModule`, but got: {model.__class__.__name__}.")

        if module:
            if module.model and not isinstance(module.model, nn.Layer) and not callable(module.model):
                raise TypeError(
                    "'model' must be sub classes of `paddle.nn.Layer` or any callable function, but got: {module.model.__class__.__name__}."
                )
            model = module.model

            if mode == "train":
                if module.loss_fn and not isinstance(module.loss_fn, nn.Layer) and not callable(module.loss_fn):
                    raise TypeError(
                        "'loss_fn' must be sub classes of `paddle.nn.Layer` or any callable function, but got: {module.loss_fn.__class__.__name__}."
                    )
            else:
                module.loss_fn = None
                module.model.eval()
            loss_fn = module.loss_fn

        self._module = module

        # global configs
        self._global_batch_size = configs["Global"]["global_batch_size"]
        self._local_batch_size = configs["Global"]["local_batch_size"]

        # Distributed
        self._pp_degree = configs["Distributed"]["pp_degree"]
        pipeline_cfg = configs.Distributed.get("pipeline", {})
        self._job_schedule_profiler_start = pipeline_cfg.get("job_schedule_profiler_start", -1)
        self._job_schedule_profiler_end = pipeline_cfg.get("job_schedule_profiler_end", -1)
        
        # engine configs
        self._configs = configs["Engine"]

        self._run_mode = self._configs.get("run_mode", "step")
        assert self._run_mode in ["epoch", "step"], "run_mode must be epoch or step"

        self._max_steps = self._configs["max_steps"]
        self._eval_freq = self._configs["eval_freq"]
        self._eval_iters = self._configs["eval_iters"]
        self._test_iters = self._configs["test_iters"]
        self._logging_freq = self._configs["logging_freq"]
        self._num_train_epochs = self._configs["num_train_epochs"]
        self._accumulate_steps = self._configs["accumulate_steps"]
        self._strategy = self._configs["strategy"]

        # save & load
        self._save_steps = self._configs["save_load"]["save_steps"]
        self._save_epoch = self._configs["save_load"]["save_epoch"]
        self._output_dir = self._configs["save_load"]["output_dir"]
        self._ckpt_dir = self._configs["save_load"]["ckpt_dir"]

        # lr_scheduler and optimizer
        if mode == "train":
            self._use_increments = configs.Optimizer.lr.pop("use_increments", False)
            self._lr_scheduler_mode = configs.Optimizer.lr.pop("run_mode", "step")
            assert self._lr_scheduler_mode in ["epoch", "step"], "lr.run_mode must be epoch or step"
        self._lr_scheduler = build_lr_scheduler(configs.Optimizer.lr) if mode == "train" else None
        self._optimizer = (
            build_optimizer(configs.Optimizer, model, self._lr_scheduler) if mode == "train" else None
        )

        # init engine
        self._auto_engine = auto.Engine(model, loss_fn, self._optimizer, strategy=self._strategy)

        # using for save/load
        self._load_recovery = {"step": 0, "epoch": 0}

        self.profiler = None
        if "Profiler" in configs and configs.get("Profiler", {}).get("enable", False):
            self.profiler_config = configs["Profiler"]

            scheduler = self.profiler_config.get("scheduler", None)
            profiler_log = self.profiler_config.get("profiler_log", "./profiler_log")
            record_shapes = self.profiler_config.get("record_shapes", True)
            profile_memory = self.profiler_config.get("profile_memory", True)
            self.profiler = paddle.profiler.Profiler(
                targets=[paddle.profiler.ProfilerTarget.CPU, paddle.profiler.ProfilerTarget.GPU],
                scheduler=scheduler,
                on_trace_ready=paddle.profiler.export_chrome_tracing(profiler_log),
                record_shapes=record_shapes,
                profile_memory=profile_memory,
            )
            self.profiler.start()
            logger.warning("Profiler is enabled, do not enable it in production.")

        # Profiler_auto configs
        self.memory_stats = configs.get("Profiler_auto", {}).get("memory_stats", False)
        self.nvprof_start = configs.get("Profiler_auto", {}).get("nvprof_start", -1)
        self.nvprof_end = configs.get("Profiler_auto", {}).get("nvprof_end", -1)
        
        if (self._job_schedule_profiler_start != -1) and use_new_executor():
            logger.info("Schedule Profiler start at step {} and end at step {}".format(self._job_schedule_profiler_start, self._job_schedule_profiler_end))

    def _validate_batch(self, batch):
        if self._pp_degree > 1 or self._accumulate_steps == 1:
            batches = batch
        else:
            feed_names = []
            split_batches = []
            for n, b in batch[0].items():
                feed_names.append(n)
                split_batches.append(np.split(np.array(b), self._accumulate_steps, 0))
            batches = []
            for i in range(len(split_batches[0])):
                micro_batch = [split_batch[i] for split_batch in split_batches]
                batches.append(dict(zip(feed_names, micro_batch)))
        return batches

    def _train_one_epoch(self, epoch_index, train_data_loader=None, valid_data_loader=None):

        train_losses = []
        train_step_start = get_timestamp()
        skip_first = True

        total_train_batch = self._max_steps if self._run_mode == "step" else len(train_data_loader)
        total_train_step = self._max_steps if self._run_mode == "step" else total_train_batch * self._num_train_epochs
        if use_new_executor():
            total_eval_batch = len(valid_data_loader) if valid_data_loader is not None else 0
        else:
            total_eval_batch = valid_data_loader._steps if valid_data_loader is not None else 0
        valid_data_loader = valid_data_loader if valid_data_loader is not None else None
        eval_finished_step = 0

        self._auto_engine.prepare(mode="train")

        for step, batch in enumerate(train_data_loader):
            with job_schedule_profiler_range(step, self._job_schedule_profiler_start, self._auto_engine.enable_job_schedule_profiler) as status:
                self._auto_engine.enable_job_schedule_profiler = status

            if epoch_index == self._load_recovery["epoch"]:
                if step < self._load_recovery["step"]:
                    continue


            fetch_list = None
            if self._strategy.amp.enable:
                # fetch_list = ["find_infinite_scale.tmp_0", "loss_scaling_0"]
                fetch_list = []

            final_loss = None
            if use_new_executor():
                batches = self._validate_batch(batch)
                for micro_batch in batches:
                    with paddle.profiler.utils._nvprof_range(iter_id=step, start=self.nvprof_start, end=self.nvprof_end):
                        outs = self._auto_engine.run(micro_batch, fetch_list=fetch_list, mode="train")
                    # pp: some devices don't have loss in outs
                    if "loss" in outs:
                        if final_loss is None:
                            final_loss = np.sum(outs["loss"])
                        else:
                            final_loss += np.sum(outs["loss"])

                if final_loss is not None and self._accumulate_steps > 1:
                    final_loss /= self._accumulate_steps
            else:
                if self._pp_degree == 1 and self._accumulate_steps > 1:  # gradient merge
                    local_steps = self._accumulate_steps
                else:
                    local_steps = 1
                for _ in range(local_steps):
                    with paddle.profiler.utils._nvprof_range(iter_id=step, start=self.nvprof_start, end=self.nvprof_end):
                        outs = self._auto_engine.run(batch, fetch_list=fetch_list, mode="train")
                    # pp: some devices don't have loss in outs
                    if "loss" in outs:
                        if final_loss is None:
                            final_loss = np.sum(outs["loss"])
                        else:
                            final_loss += np.sum(outs["loss"])

            if final_loss is not None:
                train_losses.append(final_loss)

            if self._lr_scheduler is not None and self._lr_scheduler_mode == "step":
                self._auto_engine.optimizer._learning_rate.step(epoch=self._global_batch_size if self._use_increments else None)

            if (step + 1) % self._logging_freq == 0:
                train_step_cost = get_timestamp() - train_step_start
                numpy_losses = [float(loss) for loss in train_losses]
                log_dict = {
                    "epoch": epoch_index,
                    "total_epoch": self._num_train_epochs,
                    "batch": step,
                    "total_batch": total_train_batch,
                    "total_step": total_train_step,
                    "train_cost": train_step_cost if step == 0 else train_step_cost / self._logging_freq,
                    "lr": self._auto_engine.optimizer.get_lr(),
                    "found_inf": 0, # if self._strategy.amp.enable outs["fetches"]["find_infinite_scale.tmp_0"]
                    "dp_world_size": self._auto_engine._dp_world_sizes[0]
                }
                if len(train_losses) > 0:
                    log_dict["loss"] = sum(numpy_losses) / len(numpy_losses)
                if self._strategy.amp.enable:
                    log_dict["loss_scale"] = self._strategy.amp.init_loss_scaling  # outs["fetches"]["loss_scaling_0"]
                if self.memory_stats:
                    # convert from Byte to MB
                    log_dict["max_memory_allocated"] = paddle.device.cuda.max_memory_allocated() / (1024**2)
                    log_dict["max_memory_reserved"] = paddle.device.cuda.max_memory_reserved() / (1024**2)
                    log_dict["memory_allocated"] = paddle.device.cuda.memory_allocated() / (1024**2)
                    log_dict["memory_reserved"] = paddle.device.cuda.memory_reserved() / (1024**2)
                self._module.training_step_end(log_dict)

                train_step_start = get_timestamp()
                train_losses = []

            if self._run_mode == "step" and not skip_first:
                if self._eval_freq > 0 and step % self._eval_freq == 0:

                    eval_losses = []
                    eval_step_start = get_timestamp()

                    for eval_step, batch in enumerate(valid_data_loader):
                        eval_finished_step += 1
                        outs = self._auto_engine.run(batch, mode="eval")
                        if "loss" in outs:
                            eval_losses.append(outs["loss"])

                        if eval_step >= self._eval_iters - 1:
                            break

                    numpy_losses = [float(loss) for loss in eval_losses]
                    eval_step_cost = get_timestamp() - eval_step_start

                    log_dict = {
                        "epoch": epoch_index,
                        "batch": eval_finished_step,
                        "total_batch": total_eval_batch,
                        "eval_cost": eval_step_cost / self._logging_freq,
                    }
                    if len(eval_losses) > 0:
                        log_dict["loss"] = sum(numpy_losses) / len(numpy_losses)
                    self._module.validation_step_end(log_dict)

                if self._save_steps > 0 and step % self._save_steps == 0:
                    device_synchronize()
                    self.save(epoch=epoch_index, step=step)
            else:
                skip_first = False

            if self._run_mode == "step" and step >= self._max_steps:
                return

            if self.profiler:
                self.profiler.step()

    def fit(self, epoch=1, train_dataset=None, valid_dataset=None):

        train_start = get_timestamp()

        start_epoch = self._load_recovery["epoch"]

        train_data_loader, valid_data_loader = None, None
        if train_dataset:
            if use_new_executor():
                train_data_loader = self._auto_engine.dataloader(
                    dataset=train_dataset,
                    batch_size=self._global_batch_size,
                    steps_per_epoch=self._max_steps,
                    epochs=self._num_train_epochs,
                    collate_fn=train_dataset.collate_fn,
                    num_workers=1,
                    sample_split=train_dataset.sample_split,
                    mode="train",
                )
            else:
                train_data_loader = self._auto_engine.dataloader_from_generator(
                    dataset=train_dataset,
                    batch_size=self._global_batch_size,
                    steps_per_epoch=self._max_steps,
                    epochs=self._num_train_epochs,
                    collate_fn=train_dataset.collate_fn,
                    sample_split=train_dataset.sample_split,
                    mode="train",
                )
        if valid_dataset and self._eval_freq <= self._max_steps:
            if use_new_executor():
                valid_data_loader = self._auto_engine.dataloader(
                    dataset=valid_dataset,
                    batch_size=self._global_batch_size,
                    steps_per_epoch=self._max_steps,
                    epochs=self._num_train_epochs,
                    collate_fn=valid_dataset.collate_fn,
                    num_workers=1,
                    sample_split=valid_dataset.sample_split,
                    mode="eval",
                )
            else:
                valid_data_loader = self._auto_engine.dataloader_from_generator(
                    dataset=valid_dataset,
                    batch_size=self._global_batch_size,
                    steps_per_epoch=self._max_steps,
                    epochs=self._num_train_epochs,
                    collate_fn=valid_dataset.collate_fn,
                    sample_split=valid_dataset.sample_split,
                    mode="eval",
                )

        for epoch_index in range(start_epoch, epoch):
            train_epoch_start = get_timestamp()
            self._train_one_epoch(epoch_index, train_data_loader, valid_data_loader)
            train_epoch_cost = get_timestamp() - train_epoch_start
            log_dict = {
                "epoch": epoch_index,
                "train_cost": train_epoch_cost,
            }
            self._module.training_epoch_end(log_dict)

            if self._lr_scheduler is not None and self._lr_scheduler_mode == "epoch":
                self._lr_scheduler.step()

            if self._run_mode == "epoch" and self._eval_freq > 0 and epoch_index % self._eval_freq == 0:
                eval_epoch_start = get_timestamp()
                self._evaluate_one_epoch(epoch_index, valid_data_loader)
                eval_epoch_cost = get_timestamp() - eval_epoch_start
                log_dict = {
                    "epoch": epoch_index,
                    "eval_cost": eval_epoch_cost,
                }
                self._module.validation_epoch_end(log_dict)

            if self._save_epoch > 0 and self._run_mode == "epoch" and epoch_index % self._save_epoch == 0:
                self.save(epoch=epoch_index, step=len(train_data_loader))

        logger.info(
            "The training process is complete and total cost of time for training is : {}".format(
                convert_timestamp_to_data(get_timestamp() - train_start)
            )
        )
        if valid_data_loader and not use_new_executor():
            valid_data_loader._inner_dataloader.reset()

        if self.profiler:
            self._profiler_done()

    def evaluate(self, epoch=1, valid_dataset=None):

        valid_data_loader = None
        if valid_dataset:
            if use_new_executor():
                valid_data_loader = self._auto_engine.dataloader(
                    dataset=valid_dataset,
                    batch_size=self._global_batch_size,
                    steps_per_epoch=self._max_steps,
                    epochs=self._num_train_epochs,
                    collate_fn=valid_dataset.collate_fn,
                    num_workers=1,
                    sample_split=valid_dataset.sample_split,
                    mode="eval",
                )
            else:
                valid_data_loader = self._auto_engine.dataloader_from_generator(
                    dataset=valid_dataset,
                    batch_size=self._global_batch_size,
                    steps_per_epoch=self._max_steps,
                    epochs=self._num_train_epochs,
                    collate_fn=valid_dataset.collate_fn,
                    num_workers=1,
                    sample_split=valid_dataset.sample_split,
                    mode="eval",
                )

        for epoch_index in range(epoch):
            eval_epoch_start = get_timestamp()
            self._evaluate_one_epoch(epoch_index, valid_data_loader)

            eval_epoch_cost = get_timestamp() - eval_epoch_start
            log_dict = {
                "epoch": epoch_index,
                "eval_cost": eval_epoch_cost,
            }
            self._module.validation_epoch_end(log_dict)

        logger.info("The evaluting process is complete.")
        del valid_data_loader
        return

    def _evaluate_one_epoch(self, epoch=1, valid_data_loader=None):

        eval_step_start = get_timestamp()
        eval_losses = []
        total_eval_batch = len(valid_data_loader)
        valid_data_loader = valid_data_loader() if valid_data_loader is not None else None
        for eval_step, batch in enumerate(valid_data_loader):
            with paddle.profiler.utils._nvprof_range(iter_id=eval_step, start=self.nvprof_start, end=self.nvprof_end):
                outs = self._auto_engine.run(batch, mode="eval")
            eval_losses.append(outs["loss"])

            if eval_step % self._logging_freq == 0:
                eval_losses = [float(loss) for loss in eval_losses]
                eval_step_cost = get_timestamp() - eval_step_start
                log_dict = {
                    "loss": sum(eval_losses) / len(eval_losses),
                    "epoch": epoch,
                    "batch": eval_step,
                    "total_batch": total_eval_batch,
                    "eval_cost": eval_step_cost if eval_step == 0 else eval_step_cost / self._logging_freq,
                }
                self._module.validation_step_end(log_dict)
                eval_step_start = get_timestamp()
                eval_losses = []

            if self._run_mode == "step" and eval_step >= self._max_steps:
                logger.info("[eval] epoch {} : evaluting process is complete.".
                            format(epoch))
                return

    def predict(self, epoch=1, test_dataset=None):

        test_data_loader = None
        if test_dataset:
            if use_new_executor():
                test_data_loader = self._auto_engine.dataloader(
                    dataset=test_dataset,
                    batch_size=self._global_batch_size,
                    steps_per_epoch=self._max_steps,
                    epochs=self._num_train_epochs,
                    collate_fn=test_dataset.collate_fn,
                    num_workers=1,
                    sample_split=test_dataset.sample_split,
                    mode="predict",
                )
            else:
                test_data_loader = self._auto_engine.dataloader_from_generator(
                    dataset=test_dataset,
                    batch_size=self._global_batch_size,
                    steps_per_epoch=self._max_steps,
                    epochs=self._num_train_epochs,
                    collate_fn=test_dataset.collate_fn,
                    num_workers=1,
                    sample_split=test_dataset.sample_split,
                    mode="predict",
                )

        test_start = get_timestamp()
        test_losses = []
        for test_step, batch in enumerate(test_data_loader):
            with paddle.profiler.utils._nvprof_range(iter_id=test_step, start=self.nvprof_start, end=self.nvprof_end):
                outs = self._auto_engine.run(batch, mode="predict")
            test_losses.append(outs["loss"])

            if test_step % self._logging_freq == 0:
                test_losses = [float(loss) for loss in test_losses]
                test_cost = get_timestamp() - test_start
                log_dict = {
                    "loss": sum(test_losses) / len(test_losses),
                    "epoch": epoch,
                    "batch": test_step,
                    "test_cost": test_cost if test_step == 0 else test_cost / self._logging_freq,
                }
                self._module.test_step_end(log_dict)
                test_start = get_timestamp()
                test_losses = []

            if test_step >= self._max_steps:
                logger.info("The predicting process is complete.")
                del test_data_loader
                return

    def export(self):
        self._auto_engine.prepare(self._module.input_spec(), mode="predict")
        self.save(training=False)

    def tune(self, tune_dataset=None):
        self._auto_engine._tune(tune_dataset, tune_sample_split=tune_dataset.sample_split, batch_size=self.batch_size)

    def save(self, training=True):
        if self._output_dir and isinstance(self._output_dir, str):
            path = os.path.join(self._output_dir, "auto")
            self._auto_engine.save(path, training=training)
        else:
            raise TypeError("`save` requires a valid value of `output_dir`.")

    def load(self):
        if self._ckpt_dir and isinstance(self._ckpt_dir, str):
            self._auto_engine.load(self._ckpt_dir)
        else:
            logger.warning("`load` requires a valid value of `ckpt_dir`.")

    def export_from_prog(self):
        paddle.enable_static()

        if not (self._ckpt_dir and isinstance(self._ckpt_dir, str)):
            raise ValueError("invalid ckpt_dir.")

        exe = paddle.static.Executor()

        [inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model(
            path_prefix=self._ckpt_dir, executor=exe
        )
        feed_targets = [inference_program.global_block().var(name) for name in feed_target_names]

        self._auto_engine.prepare(
            inputs=feed_targets,
            main_program=inference_program,
            startup_program=paddle.static.Program(),
            mode="predict",
        )

        model_dict = self._auto_engine.main_program.state_dict()
        for param in list(filter(lambda var: var.persistable, self._auto_engine.main_program.list_vars())):
            if param.type in [core.VarDesc.VarType.FEED_MINIBATCH, core.VarDesc.VarType.FETCH_LIST]:
                continue
            if param.dtype != model_dict[param.name]._dtype():
                model_dict[param.name] = model_dict[param.name]._as_type(param.dtype)
        self._auto_engine.main_program.set_state_dict(model_dict)

        path = os.path.join(self._output_dir, "auto_dist0")
        paddle.static.save_inference_model(
            path,
            feed_targets,
            fetch_targets,
            exe,
            program=self._auto_engine.main_program,
        )

        paddle.disable_static()

    def _print_summary(self):
        views_dict = {
            SummaryView.DeviceView: "device",
            SummaryView.OverView: "overview",
            SummaryView.ModelView: "model",
            SummaryView.DistributedView: "dist",
            SummaryView.KernelView: "kernel",
            SummaryView.OperatorView: "op",
            SummaryView.MemoryView: "mem",
            SummaryView.MemoryManipulationView: "memcpy",
            SummaryView.UDFView: "udf",
        }

        default_views = [
            SummaryView.OverView,
            SummaryView.ModelView,
            SummaryView.KernelView,
            SummaryView.OperatorView,
        ]

        def gen_views(cfg):
            # print all summary view if detailed=True
            if self.profiler_config.get("detailed", False):
                return None

            views = []
            # override default view with user defined value if detailed=False
            for view in SummaryView:
                v = self.profiler_config.get("summary", {}).get(
                    views_dict[view], None)
                if v is True or (v is None and view in default_views):
                    views.append(view)

            return views or None

        self.profiler.summary(
            sorted_by=paddle.profiler.SortedKeys.GPUTotal,
            views=gen_views(self.profiler_config))

    def _profiler_done(self):
        if not self.profiler:
            return

        logger.info("Profiler finished, prepare to print summary...")

        self.profiler.stop()

        self._print_summary()
        profiler_log = self.profiler_config.get("profiler_log",
                                                "./profiler_log")
        logger.info(
            "For more information please install visualdl and run it with following command:"
        )
        logger.info(
            "-------------------------------------------------------------------------------"
        )
        logger.info(f"visualdl --host 0.0.0.0 --logdir {profiler_log}")
        logger.info(
            "-------------------------------------------------------------------------------"
        )

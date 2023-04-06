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

import paddle
import paddle.fluid.core as core
import paddle.nn as nn
from paddle.distributed.fleet import auto
from ppfleetx.core.engine import BasicEngine
from ppfleetx.core.module import BasicModule
try:
    from ppfleetx.optims import build_lr_scheduler, build_optimizer
except Exception:
    pass
from ppfleetx.utils.log import logger
from ppfleetx.utils.version import version_check


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

        # lr_scheduler and optimizer
        lr = build_lr_scheduler(configs.Optimizer.lr) if mode == "train" else None
        optimizer = build_optimizer(configs.Optimizer, model, lr) if mode == "train" else None

        # engine configs
        self._configs = configs["Engine"]
        self._max_steps = self._configs["max_steps"]
        self._verbose = self._configs["verbose"]
        self._eval_freq = self._configs["eval_freq"]
        self._eval_iters = self._configs["eval_iters"]
        self._test_iters = self._configs["test_iters"]
        self._logging_freq = self._configs["logging_freq"]
        self._num_train_epochs = self._configs["num_train_epochs"]
        self._strategy = self._configs["strategy"]

        # save & load
        self._save_steps = self._configs["save_load"]["save_steps"]
        self._save_epoch = self._configs["save_load"]["save_epoch"]
        self._output_dir = self._configs["save_load"]["output_dir"]
        self._ckpt_dir = self._configs["save_load"]["ckpt_dir"]

        # engine fit inputs
        self.batch_size = configs["Global"]["global_batch_size"]

        # init engine
        self._auto_engine = auto.Engine(model, loss_fn, optimizer, strategy=self._strategy)

    def fit(self, epoch=1, train_dataset=None, valid_dataset=None):

        train_sample_split = train_dataset.sample_split if train_dataset else None
        valid_sample_split = valid_dataset.sample_split if valid_dataset else None

        self._auto_engine.fit(
            train_data=train_dataset,
            valid_data=valid_dataset,
            train_sample_split=train_sample_split,
            valid_sample_split=valid_sample_split,
            epochs=self._num_train_epochs,
            batch_size=self.batch_size,
            steps_per_epoch=self._max_steps,
            valid_steps=self._eval_iters,
            valid_freq=self._eval_freq,
            collate_fn=train_dataset.collate_fn,
            log_freq=self._logging_freq,
            save_dir=self._output_dir,
            save_freq=self._save_steps,
            verbose=self._verbose,
        )

    def evaluate(self, valid_dataset=None):

        self._auto_engine.evaluate(
            valid_data=valid_dataset,
            valid_sample_split=valid_dataset.sample_split,
            batch_size=self.batch_size,
            steps=self._max_steps,
            collate_fn=valid_dataset.collate_fn,
        )

    def predict(self, test_dataset=None):

        self._auto_engine.predict(
            test_data=test_dataset,
            test_sample_split=test_dataset.sample_split,
            batch_size=self.batch_size,
            steps=self._max_steps,
            collate_fn=test_dataset.collate_fn,
        )

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

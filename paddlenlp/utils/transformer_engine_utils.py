# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from contextlib import contextmanager

import paddle
from paddle.distributed import fleet

try:
    import transformer_engine.paddle as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    _IS_TRANSFORMER_ENGINE_INSTALLED = True

except ModuleNotFoundError:
    _IS_TRANSFORMER_ENGINE_INSTALLED = False


def get_filename_for_this_rank(filenames, rank):
    # split filename into 3 parts
    for filename in filenames:
        filename_split = filename.split(".")
        assert len(filename_split) == 3, "filename must be like model_state.tp01.pdparams"
        file_rank = int(filename_split[1][2:])
        # file_rank = int(filename_split[1])

        if file_rank == rank:
            return filename


def check_params_valid(state_dict, state_dict_original):
    for name, param in state_dict.items():
        if name not in state_dict_original:
            raise ValueError("The input checkpoint is not a valid checkpoint.")
        if param.shape != state_dict_original[name].shape:
            raise ValueError("The input checkpoint is not a valid checkpoint.")


class TransformerEngineHelper:
    @staticmethod
    def is_installed():
        return _IS_TRANSFORMER_ENGINE_INSTALLED

    @staticmethod
    def get_rope_layer():
        assert (
            TransformerEngineHelper.is_installed()
        ), "TransformerEngine is not installed. Please install it first or disable it."
        return te.RotaryPositionEmbedding

    @staticmethod
    def get_transformer_layer():
        assert (
            TransformerEngineHelper.is_installed()
        ), "TransformerEngine is not installed. Please install it first or disable it."
        return te.TransformerLayer

    @staticmethod
    def get_te_recompute_func():
        assert (
            TransformerEngineHelper.is_installed()
        ), "TransformerEngine is not installed. Please install it first or disable it."
        return te.recompute

    @staticmethod
    def get_fp8_group():
        assert (
            TransformerEngineHelper.is_installed()
        ), "TransformerEngine is not installed. Please install it first or disable it."

        try:
            hcg = fleet.get_hybrid_communicate_group()
        except:
            return None
        use_pp = hcg.get_pipe_parallel_world_size() > 1
        if not use_pp:
            return None

        dp_group = hcg.get_data_parallel_group()
        tp_group = hcg.get_model_parallel_group()
        if dp_group.nranks <= 1:
            return tp_group
        if tp_group.nranks <= 1:
            return dp_group

        return dp_group

    @staticmethod
    @contextmanager
    def fp8_autocast(enabled=False, fp8_group=None):
        if TransformerEngineHelper.is_installed():
            fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
            fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=1, amax_compute_algo="most_recent")
            with te.fp8_autocast(enabled=enabled, fp8_group=fp8_group, fp8_recipe=fp8_recipe):
                yield
        else:  # null context
            yield

    @staticmethod
    def te_init_weights(layer, config):
        if not TransformerEngineHelper.is_installed():
            return
        if isinstance(
            layer,
            (
                te.LayerNormLinear,
                te.Linear,
            ),
        ):
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )

        if isinstance(layer, te.LayerNormMLP):
            if isinstance(layer.fc1_weight, paddle.Tensor):
                layer.fc1_weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=config.initializer_range,
                        shape=layer.fc1_weight.shape,
                    )
                )
            if isinstance(layer.fc2_weight, paddle.Tensor):
                layer.fc2_weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=config.initializer_range,
                        shape=layer.fc2_weight.shape,
                    )
                )

    @staticmethod
    def reset_te_init_weights(trainer, ckpt_path):
        if not TransformerEngineHelper.is_installed() or ckpt_path is None:
            return

        trainer._load_from_checkpoint(resume_from_checkpoint=ckpt_path)

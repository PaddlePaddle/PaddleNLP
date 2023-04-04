# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import gc
import tempfile
import unittest

import paddle
import paddle.nn as nn
from parameterized import parameterized
from ppdiffusers_test.test_modeling_common import ModelTesterMixin

from ppdiffusers import UNet2DConditionModel
from ppdiffusers.models.cross_attention import (
    CrossAttnProcessor,
    LoRACrossAttnProcessor,
)
from ppdiffusers.utils import (
    floats_tensor,
    load_ppnlp_numpy,
    logging,
    paddle_all_close,
    require_paddle_gpu,
    slow,
)
from ppdiffusers.utils.import_utils import is_ppxformers_available

logger = logging.get_logger(__name__)


def create_lora_layers(model):
    lora_attn_procs = {}
    for name in model.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
        lora_attn_procs[name] = lora_attn_procs[name]
        with paddle.no_grad():
            lora_attn_procs[name].to_q_lora.up.weight.set_value(lora_attn_procs[name].to_q_lora.up.weight + 1)
            lora_attn_procs[name].to_k_lora.up.weight.set_value(lora_attn_procs[name].to_k_lora.up.weight + 1)
            lora_attn_procs[name].to_v_lora.up.weight.set_value(lora_attn_procs[name].to_v_lora.up.weight + 1)
            lora_attn_procs[name].to_out_lora.up.weight.set_value(lora_attn_procs[name].to_out_lora.up.weight + 1)
    return lora_attn_procs


class UNet2DConditionModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DConditionModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = 32, 32
        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor([10])
        encoder_hidden_states = floats_tensor((batch_size, 4, 32))
        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return 4, 32, 32

    @property
    def output_shape(self):
        return 4, 32, 32

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_xformers_enable_works(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.enable_xformers_memory_efficient_attention()
        assert hasattr(
            model.mid_block.attentions[0].transformer_blocks[0].attn1.processor, "attention_op"
        ), "xformers is not enabled"

    def test_gradient_checkpointing(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        assert not model.is_gradient_checkpointing and model.training
        out = model(**inputs_dict).sample
        model.clear_gradients()
        labels = paddle.randn_like(out)
        loss = (out - labels).mean()
        loss.backward()
        model_2 = self.model_class(**init_dict)
        model_2.set_state_dict(state_dict=model.state_dict())
        model_2.enable_gradient_checkpointing()
        assert model_2.is_gradient_checkpointing and model_2.training
        out_2 = model_2(**inputs_dict).sample
        model_2.clear_gradients()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()
        self.assertTrue((loss - loss_2).abs() < 1e-05)
        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())
        for name, param in named_params.items():
            self.assertTrue(paddle_all_close(param.grad, named_params_2[name].grad, atol=5e-05))

    def test_model_with_attention_head_dim_tuple(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        model = self.model_class(**init_dict)
        model.eval()
        with paddle.no_grad():
            output = model(**inputs_dict)
            if isinstance(output, dict):
                output = output.sample
        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_use_linear_projection(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["use_linear_projection"] = True
        model = self.model_class(**init_dict)
        model.eval()
        with paddle.no_grad():
            output = model(**inputs_dict)
            if isinstance(output, dict):
                output = output.sample
        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_attention_slicing(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        model = self.model_class(**init_dict)
        model.eval()
        model.set_attention_slice("auto")
        with paddle.no_grad():
            output = model(**inputs_dict)
        assert output is not None
        model.set_attention_slice("max")
        with paddle.no_grad():
            output = model(**inputs_dict)
        assert output is not None
        model.set_attention_slice(2)
        with paddle.no_grad():
            output = model(**inputs_dict)
        assert output is not None

    def test_model_slicable_head_dim(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        model = self.model_class(**init_dict)

        def check_slicable_dim_attr(module: paddle.nn.Layer):
            if hasattr(module, "set_attention_slice"):
                assert isinstance(module.sliceable_head_dim, int)
            for child in module.children():
                check_slicable_dim_attr(child)

        for module in model.children():
            check_slicable_dim_attr(module)

    def test_special_attn_proc(self):
        class AttnEasyProc(nn.Layer):
            def __init__(self, num):
                super().__init__()
                self.weight = self.create_parameter(
                    (1,), dtype=paddle.get_default_dtype(), default_initializer=nn.initializer.Constant(num)
                )
                self.is_run = False
                self.number = 0
                self.counter = 0

            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, number=None):
                batch_size, sequence_length, _ = hidden_states.shape
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                query = attn.to_q(hidden_states)
                encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)
                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                hidden_states = paddle.matmul(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                hidden_states += self.weight
                self.is_run = True
                self.counter += 1
                self.number = number
                return hidden_states

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        model = self.model_class(**init_dict)
        processor = AttnEasyProc(5.0)
        model.set_attn_processor(processor)
        model(**inputs_dict, cross_attention_kwargs={"number": 123}).sample
        assert processor.counter == 12
        assert processor.is_run
        assert processor.number == 123

    def test_lora_processors(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        model = self.model_class(**init_dict)
        with paddle.no_grad():
            sample1 = model(**inputs_dict).sample
        lora_attn_procs = {}
        for name in model.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = model.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(model.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = model.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
            with paddle.no_grad():
                lora_attn_procs[name].to_q_lora.up.weight.set_value(lora_attn_procs[name].to_q_lora.up.weight + 1)
                lora_attn_procs[name].to_k_lora.up.weight.set_value(lora_attn_procs[name].to_k_lora.up.weight + 1)
                lora_attn_procs[name].to_v_lora.up.weight.set_value(lora_attn_procs[name].to_v_lora.up.weight + 1)
                lora_attn_procs[name].to_out_lora.up.weight.set_value(lora_attn_procs[name].to_out_lora.up.weight + 1)
        model.set_attn_processor(lora_attn_procs)
        model.set_attn_processor(model.attn_processors)
        with paddle.no_grad():
            sample2 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample
            sample3 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
            sample4 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
        assert (sample1 - sample2).abs().max() < 0.0001
        assert (sample3 - sample4).abs().max() < 0.0001
        assert (sample2 - sample3).abs().max() > 0.0001

    def test_lora_save_load(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        paddle.seed(0)
        model = self.model_class(**init_dict)
        with paddle.no_grad():
            old_sample = model(**inputs_dict).sample
        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)
        with paddle.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname)
            paddle.seed(0)
            new_model = self.model_class(**init_dict)
            new_model.load_attn_procs(tmpdirname)
        with paddle.no_grad():
            new_sample = new_model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
        assert (sample - new_sample).abs().max() < 0.0001
        assert (sample - old_sample).abs().max() > 0.0001

    def test_lora_on_off(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        paddle.seed(0)
        model = self.model_class(**init_dict)
        with paddle.no_grad():
            old_sample = model(**inputs_dict).sample
        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)
        with paddle.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample
        model.set_attn_processor(CrossAttnProcessor())
        with paddle.no_grad():
            new_sample = model(**inputs_dict).sample
        assert (sample - new_sample).abs().max() < 0.0001
        assert (sample - old_sample).abs().max() < 0.0001

    @unittest.skipIf(
        not is_ppxformers_available(),
        reason="scaled_dot_product_attention attention is only available with CUDA and `scaled_dot_product_attention` installed",
    )
    def test_lora_xformers_on_off(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        paddle.seed(0)
        model = self.model_class(**init_dict)
        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)
        with paddle.no_grad():
            sample = model(**inputs_dict).sample
            model.enable_xformers_memory_efficient_attention()
            on_sample = model(**inputs_dict).sample
            model.disable_xformers_memory_efficient_attention()
            off_sample = model(**inputs_dict).sample
        assert (sample - on_sample).abs().max() < 0.05
        assert (sample - off_sample).abs().max() < 0.05


@slow
class UNet2DConditionModelIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_latents(self, seed=0, shape=(4, 4, 64, 64), fp16=False):
        dtype = paddle.float16 if fp16 else paddle.float32
        image = paddle.to_tensor(data=load_ppnlp_numpy(self.get_file_format(seed, shape))).cast(dtype)
        return image

    def get_unet_model(self, fp16=False, model_id="CompVis/stable-diffusion-v1-4"):
        revision = "fp16" if fp16 else None
        paddle_dtype = paddle.float16 if fp16 else paddle.float32
        model = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", paddle_dtype=paddle_dtype, revision=revision
        )
        model.eval()
        return model

    def test_set_attention_slice_auto(self):
        paddle.device.cuda.empty_cache()
        unet = self.get_unet_model()
        unet.set_attention_slice("auto")
        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1
        with paddle.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        mem_bytes = paddle.device.cuda.memory_allocated()
        assert mem_bytes < 5 * 10**9

    def test_set_attention_slice_max(self):
        paddle.device.cuda.empty_cache()
        unet = self.get_unet_model()
        unet.set_attention_slice("max")
        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1
        with paddle.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        mem_bytes = paddle.device.cuda.memory_allocated()
        assert mem_bytes < 5 * 10**9

    def test_set_attention_slice_int(self):
        paddle.device.cuda.empty_cache()
        unet = self.get_unet_model()
        unet.set_attention_slice(2)
        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1
        with paddle.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        mem_bytes = paddle.device.cuda.memory_allocated()
        assert mem_bytes < 5 * 10**9

    def test_set_attention_slice_list(self):
        paddle.device.cuda.empty_cache()
        slice_list = 16 * [2, 3]
        unet = self.get_unet_model()
        unet.set_attention_slice(slice_list)
        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1
        with paddle.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        mem_bytes = paddle.device.cuda.memory_allocated()
        assert mem_bytes < 5 * 10**9

    def get_encoder_hidden_states(self, seed=0, shape=(4, 77, 768), fp16=False):
        dtype = "float16" if fp16 else "float32"
        hidden_states = paddle.to_tensor(data=load_ppnlp_numpy(self.get_file_format(seed, shape))).cast(dtype)
        return hidden_states

    @parameterized.expand(
        [
            [33, 4, [-0.4424, 0.151, -0.1937, 0.2118, 0.3746, -0.3957, 0.016, -0.0435]],
            [47, 0.55, [-0.1508, 0.0379, -0.3075, 0.254, 0.3633, -0.0821, 0.1719, -0.0207]],
            [21, 0.89, [-0.6479, 0.6364, -0.3464, 0.8697, 0.4443, -0.6289, -0.0091, 0.1778]],
            [9, 1000, [0.8888, -0.5659, 0.5834, -0.7469, 1.1912, -0.3923, 1.1241, -0.4424]],
        ]
    )
    @require_paddle_gpu
    def test_compvis_sd_v1_4(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)
        timestep = paddle.to_tensor([timestep], dtype="int64")
        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        assert sample.shape == latents.shape
        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.01)

    @parameterized.expand(
        [
            [83, 4, [-0.2323, -0.1304, 0.0813, -0.3093, -0.0919, -0.1571, -0.1125, -0.5806]],
            [17, 0.55, [-0.0831, -0.2443, 0.0901, -0.0919, 0.3396, 0.0103, -0.3743, 0.0701]],
            [8, 0.89, [-0.4863, 0.0859, 0.0875, -0.1658, 0.9199, -0.0114, 0.4839, 0.4639]],
            [3, 1000, [-0.5649, 0.2402, -0.5518, 0.1248, 1.1328, -0.2443, -0.0325, -1.0078]],
        ]
    )
    @require_paddle_gpu
    def test_compvis_sd_v1_4_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4", fp16=True)
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)
        timestep = paddle.to_tensor([timestep], dtype="int64")
        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        assert sample.shape == latents.shape
        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.005)

    @parameterized.expand(
        [
            [33, 4, [-0.443, 0.157, -0.1867, 0.2376, 0.3205, -0.3681, 0.0525, -0.0722]],
            [47, 0.55, [-0.1415, 0.0129, -0.3136, 0.2257, 0.343, -0.0536, 0.2114, -0.0436]],
            [21, 0.89, [-0.7091, 0.6664, -0.3643, 0.9032, 0.4499, -0.6541, 0.0139, 0.175]],
            [9, 1000, [0.8878, -0.5659, 0.5844, -0.7442, 1.1883, -0.3927, 1.1192, -0.4423]],
        ]
    )
    @require_paddle_gpu
    def test_compvis_sd_v1_5(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-v1-5")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)
        timestep = paddle.to_tensor([timestep], dtype="int64")
        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        assert sample.shape == latents.shape
        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.01)

    @parameterized.expand(
        [
            [83, 4, [-0.2695, -0.1669, 0.0073, -0.3181, -0.1187, -0.1676, -0.1395, -0.5972]],
            [17, 0.55, [-0.129, -0.2588, 0.0551, -0.0916, 0.3286, 0.0238, -0.3669, 0.0322]],
            [8, 0.89, [-0.5283, 0.1198, 0.087, -0.1141, 0.9189, -0.015, 0.5474, 0.4319]],
            [3, 1000, [-0.5601, 0.2411, -0.5435, 0.1268, 1.1338, -0.2427, -0.028, -1.002]],
        ]
    )
    @require_paddle_gpu
    def test_compvis_sd_v1_5_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-v1-5", fp16=True)
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)
        timestep = paddle.to_tensor([timestep], dtype="int64")
        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        assert sample.shape == latents.shape
        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.005)

    @parameterized.expand(
        [
            [33, 4, [-0.7639, 0.0106, -0.1615, -0.3487, -0.0423, -0.7972, 0.0085, -0.4858]],
            [47, 0.55, [-0.6564, 0.0795, -1.9026, -0.6258, 1.8235, 1.2056, 1.2169, 0.9073]],
            [21, 0.89, [0.0327, 0.4399, -0.6358, 0.3417, 0.412, -0.5621, -0.0397, -1.043]],
            [9, 1000, [0.16, 0.7303, -1.0556, -0.3515, -0.744, -1.2037, -1.8149, -1.8931]],
        ]
    )
    @require_paddle_gpu
    def test_compvis_sd_inpaint(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-inpainting")
        latents = self.get_latents(seed, shape=(4, 9, 64, 64))
        encoder_hidden_states = self.get_encoder_hidden_states(seed)
        timestep = paddle.to_tensor([timestep], dtype="int64")
        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        assert sample.shape == [4, 4, 64, 64]
        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.01)

    @parameterized.expand(
        [
            [83, 4, [-0.1047, -1.7227, 0.1067, 0.0164, -0.5698, -0.4172, -0.1388, 1.1387]],
            [17, 0.55, [0.0975, -0.2856, -0.3508, -0.46, 0.3376, 0.293, -0.2747, -0.7026]],
            [8, 0.89, [-0.0952, 0.0183, -0.5825, -0.1981, 0.1131, 0.4668, -0.0395, -0.3486]],
            [3, 1000, [0.479, 0.4949, -1.0732, -0.7158, 0.7959, -0.9478, 0.1105, -0.9741]],
        ]
    )
    @require_paddle_gpu
    def test_compvis_sd_inpaint_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-inpainting", fp16=True)
        latents = self.get_latents(seed, shape=(4, 9, 64, 64), fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)
        timestep = paddle.to_tensor([timestep], dtype="int64")
        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        assert sample.shape == [4, 4, 64, 64]
        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.005)

    @parameterized.expand(
        [
            [83, 4, [0.1514, 0.0807, 0.1624, 0.1016, -0.1896, 0.0263, 0.0677, 0.231]],
            [17, 0.55, [0.1164, -0.0216, 0.017, 0.1589, -0.312, 0.1005, -0.0581, -0.1458]],
            [8, 0.89, [-0.1758, -0.0169, 0.1004, -0.1411, 0.1312, 0.1103, -0.1996, 0.2139]],
            [3, 1000, [0.1214, 0.0352, -0.0731, -0.1562, -0.0994, -0.0906, -0.234, -0.0539]],
        ]
    )
    @require_paddle_gpu
    def test_stabilityai_sd_v2_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="stabilityai/stable-diffusion-2", fp16=True)
        latents = self.get_latents(seed, shape=(4, 4, 96, 96), fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, shape=(4, 77, 1024), fp16=True)
        timestep = paddle.to_tensor([timestep], dtype="int64")
        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        assert sample.shape == latents.shape
        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = paddle.to_tensor(expected_slice)
        assert paddle_all_close(output_slice, expected_output_slice, atol=0.005)

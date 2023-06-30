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
# noqa F403
import unittest

from ppdiffusers.models.unet_2d_blocks import (
    AttnDownBlock2D,
    AttnDownEncoderBlock2D,
    AttnSkipDownBlock2D,
    AttnSkipUpBlock2D,
    AttnUpBlock2D,
    AttnUpDecoderBlock2D,
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    DownEncoderBlock2D,
    ResnetDownsampleBlock2D,
    ResnetUpsampleBlock2D,
    SimpleCrossAttnDownBlock2D,
    SimpleCrossAttnUpBlock2D,
    SkipDownBlock2D,
    SkipUpBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    UpBlock2D,
    UpDecoderBlock2D,
)

from .test_unet_blocks_common import UNetBlockTesterMixin


class DownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = DownBlock2D
    block_type = "down"

    def test_output(self):
        expected_slice = [
            1.4686200618743896,
            -1.0339399576187134,
            -0.6087006330490112,
            -0.9044048190116882,
            0.21288111805915833,
            -0.8680574297904968,
            -0.4164941906929016,
            -1.6082428693771362,
            -1.5554661750793457,
        ]
        super().test_output(expected_slice)


class ResnetDownsampleBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = ResnetDownsampleBlock2D
    block_type = "down"

    def test_output(self):
        expected_slice = [
            0.1373986005783081,
            -0.06267327070236206,
            0.6338546276092529,
            0.9961339235305786,
            0.012131750583648682,
            0.2271430492401123,
            0.4698519706726074,
            -1.2050957679748535,
            -0.12423264980316162,
        ]
        super().test_output(expected_slice)


class AttnDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnDownBlock2D
    block_type = "down"

    def test_output(self):
        expected_slice = [
            -3.9491326808929443,
            -0.5726033449172974,
            -0.1606975793838501,
            0.16732816398143768,
            0.480291485786438,
            -0.6275963187217712,
            0.8580896258354187,
            -2.3375632762908936,
            -1.4645881652832031,
        ]
        super().test_output(expected_slice)


class CrossAttnDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = CrossAttnDownBlock2D
    block_type = "down"

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            2.6956636905670166,
            -4.308715343475342,
            1.5738945007324219,
            0.9817700982093811,
            -2.193608283996582,
            -0.42364418506622314,
            6.60827112197876,
            0.9649910926818848,
            2.8010499477386475,
        ]
        super().test_output(expected_slice)


class SimpleCrossAttnDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SimpleCrossAttnDownBlock2D
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_encoder_hidden_states=True)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            -1.6289970874786377,
            1.3748600482940674,
            -0.10375875234603882,
            0.9955897331237793,
            -0.8343256115913391,
            0.382874071598053,
            -0.10101768374443054,
            -0.250579297542572,
            -0.9541524648666382,
        ]
        super().test_output(expected_slice)


class SkipDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SkipDownBlock2D
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_skip_sample=True)

    def test_output(self):
        expected_slice = [
            0.2892754375934601,
            -0.4464714229106903,
            -0.18036654591560364,
            -0.4965817928314209,
            -0.050021037459373474,
            -0.6248312592506409,
            -0.5183243751525879,
            -0.02524399757385254,
            0.1424381136894226,
        ]
        super().test_output(expected_slice)


class AttnSkipDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnSkipDownBlock2D
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_skip_sample=True)

    def test_output(self):
        expected_slice = [
            -0.4862610697746277,
            0.8827285766601562,
            0.7600707411766052,
            1.828415870666504,
            0.7132594585418701,
            -0.12354043126106262,
            0.7799923419952393,
            -0.2145882546901703,
            -1.3009073734283447,
        ]
        super().test_output(expected_slice)


class DownEncoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = DownEncoderBlock2D
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_temb=False)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {"in_channels": 32, "out_channels": 32}
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            2.2016096115112305,
            -0.15662731230258942,
            1.789330005645752,
            0.392975389957428,
            -4.444106578826904,
            2.293689489364624,
            -0.7877296805381775,
            0.5266609191894531,
            -0.15173353254795074,
        ]
        super().test_output(expected_slice)


class AttnDownEncoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnDownEncoderBlock2D
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_temb=False)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {"in_channels": 32, "out_channels": 32}
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            2.127671957015991,
            -0.11142143607139587,
            1.2964460849761963,
            3.6022450923919678,
            -1.7154743671417236,
            1.6823889017105103,
            -1.6448723077774048,
            -0.4970707595348358,
            -3.637833833694458,
        ]
        super().test_output(expected_slice)


class UNetMidBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UNetMidBlock2D
    block_type = "mid"

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {"in_channels": 32, "temb_channels": 128}
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            -2.115619421005249,
            -0.18567246198654175,
            -1.673149585723877,
            -0.8526121973991394,
            -0.09890538454055786,
            -2.894134998321533,
            -0.2579667568206787,
            0.02939319610595703,
            1.1619269847869873,
        ]
        super().test_output(expected_slice)


class UNetMidBlock2DCrossAttnTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UNetMidBlock2DCrossAttn
    block_type = "mid"

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            -2.235785961151123,
            -2.2744078636169434,
            0.22076213359832764,
            -3.0804693698883057,
            -1.8690654039382935,
            -4.610274791717529,
            -0.625274121761322,
            0.4143417179584503,
            -1.8598196506500244,
        ]
        super().test_output(expected_slice)


class UNetMidBlock2DSimpleCrossAttnTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UNetMidBlock2DSimpleCrossAttn
    block_type = "mid"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_encoder_hidden_states=True)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            -3.61512899,
            0.17301944,
            -0.69105405,
            -1.40025711,
            -1.59702873,
            -1.47273242,
            -0.79226393,
            -1.22910488,
            1.09667253,
        ]
        super().test_output(expected_slice)


class UpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UpBlock2D
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def test_output(self):
        expected_slice = [
            -4.957080364227295,
            0.49701011180877686,
            4.326162815093994,
            -2.624238967895508,
            1.4365060329437256,
            3.467172145843506,
            0.8403439521789551,
            1.941118597984314,
            -0.4804985523223877,
        ]
        super().test_output(expected_slice)


class ResnetUpsampleBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = ResnetUpsampleBlock2D
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def test_output(self):
        expected_slice = [
            -2.075526714324951,
            -3.90122652053833,
            -3.0005340576171875,
            -0.9611822366714478,
            -1.0546646118164062,
            -1.7606399059295654,
            -0.24509593844413757,
            -0.025167375802993774,
            -0.7591105699539185,
        ]
        super().test_output(expected_slice)


class CrossAttnUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = CrossAttnUpBlock2D
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            -1.2535507678985596,
            -2.480539083480835,
            -3.7073025703430176,
            -2.2757019996643066,
            -3.044628143310547,
            -2.0491058826446533,
            0.8988063335418701,
            0.9877803325653076,
            1.679555892944336,
        ]
        super().test_output(expected_slice)


class SimpleCrossAttnUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SimpleCrossAttnUpBlock2D
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True, include_encoder_hidden_states=True)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            -0.2477731704711914,
            -2.644524097442627,
            -2.698854684829712,
            -0.1323309689760208,
            -1.104975700378418,
            -0.9408857822418213,
            -0.05827316641807556,
            -0.3523079752922058,
            -0.8070091009140015,
        ]
        super().test_output(expected_slice)


class AttnUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnUpBlock2D
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def test_output(self):
        expected_slice = [
            -1.8902320861816406,
            -1.3337427377700806,
            -0.8851560354232788,
            1.4004807472229004,
            -0.6870196461677551,
            -1.4291317462921143,
            1.4414796829223633,
            0.6205850839614868,
            -0.7466438412666321,
        ]
        super().test_output(expected_slice)


class SkipUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SkipUpBlock2D
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def test_output(self):
        expected_slice = [
            -0.987883985042572,
            -0.5670157074928284,
            -0.6942511796951294,
            -1.0125863552093506,
            -0.605157732963562,
            -0.8832322955131531,
            -0.9034348726272583,
            -0.7994486689567566,
            -0.9313756227493286,
        ]
        super().test_output(expected_slice)


class AttnSkipUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnSkipUpBlock2D
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def test_output(self):
        expected_slice = [
            0.5064516067504883,
            0.582533061504364,
            0.7436902523040771,
            0.6235701441764832,
            -0.03481818363070488,
            -0.1513846069574356,
            -0.40579983592033386,
            -0.9227585196495056,
            -0.9879465699195862,
        ]
        super().test_output(expected_slice)


class UpDecoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UpDecoderBlock2D
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_temb=False)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {"in_channels": 32, "out_channels": 32}
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            -0.14693844318389893,
            0.4114452600479126,
            1.3881545066833496,
            0.6828031539916992,
            0.21913594007492065,
            0.9397234320640564,
            0.8490088582038879,
            -0.9372509121894836,
            -0.16005855798721313,
        ]
        super().test_output(expected_slice)


class AttnUpDecoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnUpDecoderBlock2D
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_temb=False)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {"in_channels": 32, "out_channels": 32}
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [
            -1.6499664783477783,
            -2.1455278396606445,
            -1.504562497138977,
            -2.667104482650757,
            -3.483185291290283,
            -2.0631113052368164,
            0.9261775612831116,
            -0.60399329662323,
            -0.1882866621017456,
        ]
        super().test_output(expected_slice)

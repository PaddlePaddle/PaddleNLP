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

import paddle
from annotator.mlsd import utils


class BlockTypeA(paddle.nn.Layer):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale=True):
        super(BlockTypeA, self).__init__()
        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=in_c2, out_channels=out_c2, kernel_size=1),
            paddle.nn.BatchNorm2D(
                num_features=out_c2,
                momentum=1 - 0.1,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
                use_global_stats=True,
            ),
            paddle.nn.ReLU(),
        )
        self.conv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=in_c1, out_channels=out_c1, kernel_size=1),
            paddle.nn.BatchNorm2D(
                num_features=out_c1,
                momentum=1 - 0.1,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
                use_global_stats=True,
            ),
            paddle.nn.ReLU(),
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = paddle.nn.functional.interpolate(x=b, scale_factor=2.0, mode="bilinear", align_corners=True)
        return paddle.concat(x=(a, b), axis=1)


class BlockTypeB(paddle.nn.Layer):
    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(
                num_features=in_c,
                momentum=1 - 0.1,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
                use_global_stats=True,
            ),
            paddle.nn.ReLU(),
        )
        self.conv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(
                num_features=out_c,
                momentum=1 - 0.1,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
                use_global_stats=True,
            ),
            paddle.nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x


class BlockTypeC(paddle.nn.Layer):
    def __init__(self, in_c, out_c):
        super(BlockTypeC, self).__init__()
        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=5, dilation=5),
            paddle.nn.BatchNorm2D(
                num_features=in_c,
                momentum=1 - 0.1,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
                use_global_stats=True,
            ),
            paddle.nn.ReLU(),
        )
        self.conv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(
                num_features=in_c,
                momentum=1 - 0.1,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
                use_global_stats=True,
            ),
            paddle.nn.ReLU(),
        )
        self.conv3 = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(paddle.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.channel_pad = out_planes - in_planes
        self.stride = stride
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            paddle.nn.Conv2D(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(
                num_features=out_planes,
                momentum=1 - 0.1,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
                use_global_stats=True,
            ),
            paddle.nn.ReLU6(),
        )
        self.max_pool = paddle.nn.MaxPool2D(kernel_size=stride, stride=stride)

    def forward(self, x):
        if self.stride == 2:
            x = paddle.nn.functional.pad(x=x, pad=(0, 1, 0, 1), mode="constant", value=0)
        for module in self:
            if not isinstance(module, paddle.nn.MaxPool2D):
                x = module(x)
        return x


class InvertedResidual(paddle.nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                paddle.nn.Conv2D(
                    in_channels=hidden_dim, out_channels=oup, kernel_size=1, stride=1, padding=0, bias_attr=False
                ),
                paddle.nn.BatchNorm2D(
                    num_features=oup,
                    momentum=1 - 0.1,
                    epsilon=1e-05,
                    weight_attr=None,
                    bias_attr=None,
                    use_global_stats=True,
                ),
            ]
        )
        self.conv = paddle.nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(paddle.nn.Layer):
    def __init__(self):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        width_mult = 1.0
        round_nearest = 8
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                "inverted_residual_setting should be non-empty or a 4-element list, got {}".format(
                    inverted_residual_setting
                )
            )
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(4, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = paddle.nn.Sequential(*features)
        self.fpn_selected = [1, 3, 6, 10, 13]
        for m in self.named_sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                utils.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    utils.zeros_(m.bias)
            elif isinstance(m, paddle.nn.BatchNorm2D):
                utils.ones_(m.weight)
                utils.zeros_(m.bias)
            elif isinstance(m, paddle.nn.Linear):
                utils.normal_(m.weight, 0, 0.01)
                utils.zeros_(m.bias)

    def _forward_impl(self, x):
        fpn_features = []
        for i, f in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)
        c1, c2, c3, c4, c5 = fpn_features
        return c1, c2, c3, c4, c5

    def forward(self, x):
        return self._forward_impl(x)


class MobileV2_MLSD_Large(paddle.nn.Layer):
    def __init__(self):
        super(MobileV2_MLSD_Large, self).__init__()
        self.backbone = MobileNetV2()
        self.block15 = BlockTypeA(in_c1=64, in_c2=96, out_c1=64, out_c2=64, upscale=False)
        self.block16 = BlockTypeB(128, 64)
        self.block17 = BlockTypeA(in_c1=32, in_c2=64, out_c1=64, out_c2=64)
        self.block18 = BlockTypeB(128, 64)
        self.block19 = BlockTypeA(in_c1=24, in_c2=64, out_c1=64, out_c2=64)
        self.block20 = BlockTypeB(128, 64)
        self.block21 = BlockTypeA(in_c1=16, in_c2=64, out_c1=64, out_c2=64)
        self.block22 = BlockTypeB(128, 64)
        self.block23 = BlockTypeC(64, 16)
        print("MobileV2_MLSD_Large: ", MobileNetV2, self.backbone)

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.backbone(x)
        x = self.block15(c4, c5)
        x = self.block16(x)
        x = self.block17(c3, x)
        x = self.block18(x)
        x = self.block19(c2, x)
        x = self.block20(x)
        x = self.block21(c1, x)
        x = self.block22(x)
        x = self.block23(x)
        x = x[:, 7:, :, :]
        return x

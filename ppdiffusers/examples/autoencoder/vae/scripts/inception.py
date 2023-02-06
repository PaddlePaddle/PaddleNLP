# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) mseitzer Author. All Rights Reserved.
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
from paddle.utils.download import get_weights_path_from_url

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = (
    "https://paddlenlp.bj.bcebos.com/models/mseitzer/pp_inception-2015-12-05-6726825d.pdparams",
    "8e2ae24c34c5c8b81d45167bb9361f4c",
)
WEIGHTS_PATH = "pp_inception-2015-12-05-6726825d.pdparams"


class ConvNormActivation(nn.Sequential):
    """
    Configurable block used for Convolution-Normalzation-Activation blocks.
    This code is based on the torchvision code with modifications.
    You can also see at https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L68
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
        kernel_size: (int|list|tuple, optional): Size of the convolving kernel. Default: 3
        stride (int|list|tuple, optional): Stride of the convolution. Default: 1
        padding (int|str|tuple|list, optional): Padding added to all four sides of the input. Default: None,
            in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., paddle.nn.Layer], optional): Norm layer that will be stacked on top of the convolutiuon layer.
            If ``None`` this layer wont be used. Default: ``paddle.nn.BatchNorm2D``
        activation_layer (Callable[..., paddle.nn.Layer], optional): Activation function which will be stacked on top of the normalization
            layer (if not ``None``), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``paddle.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        norm_layer=nn.BatchNorm2D,
        activation_layer=nn.ReLU,
        dilation=1,
        bias=None,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias_attr=bias,
            )
        ]
        if norm_layer is not None:
            # The hyperparameter of BatchNorm2D is different from PaddlePaddle.
            layers.append(norm_layer(out_channels, momentum=0.1, epsilon=0.001))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class InceptionV3(nn.Layer):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3,  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    ):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in paddle.vision. The FID Inception model has different
            weights and a slightly different structure from paddle.vision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "Last possible output block index is 3"

        self.blocks = nn.LayerList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.inception_stem.conv_1a_3x3,
            inception.inception_stem.conv_2a_3x3,
            inception.inception_stem.conv_2b_3x3,
            inception.inception_stem.max_pool,
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.inception_stem.conv_3b_1x1,
                inception.inception_stem.conv_4a_3x3,
                inception.inception_stem.max_pool,
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.inception_block_list[0],
                inception.inception_block_list[1],
                inception.inception_block_list[2],
                inception.inception_block_list[3],
                inception.inception_block_list[4],
                inception.inception_block_list[5],
                inception.inception_block_list[6],
                inception.inception_block_list[7],
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.inception_block_list[8],
                inception.inception_block_list[9],
                inception.inception_block_list[10],
                inception.avg_pool,
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.stop_gradient = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : paddle.Tensor
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of paddle.Tensor, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def hack_bn_layer(layer):
    if isinstance(layer, nn.BatchNorm2D):
        layer._momentum = 0.1
        layer._epsilon = 0.001


def _inception_v3(*args, **kwargs):
    """Wraps `paddle.vision.models.inception_v3`"""
    return paddle.vision.models.inception_v3(*args, **kwargs).apply(hack_bn_layer)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than paddle.vision's Inception.

    This method first constructs paddle.vision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, with_pool=True, pretrained=False)
    inception.inception_block_list[0] = InceptionA(192, pool_features=32)
    inception.inception_block_list[1] = InceptionA(256, pool_features=64)
    inception.inception_block_list[2] = InceptionA(288, pool_features=64)
    inception.inception_block_list[4] = InceptionC(768, channels_7x7=128)
    inception.inception_block_list[5] = InceptionC(768, channels_7x7=160)
    inception.inception_block_list[6] = InceptionC(768, channels_7x7=160)
    inception.inception_block_list[7] = InceptionC(768, channels_7x7=192)
    inception.inception_block_list[9] = InceptionE_1(1280)
    inception.inception_block_list[10] = InceptionE_2(2048)

    weight_path = get_weights_path_from_url(FID_WEIGHTS_URL[0], FID_WEIGHTS_URL[1])
    state_dict = paddle.load(weight_path)
    inception.set_state_dict(state_dict)
    return inception


class InceptionA(nn.Layer):
    def __init__(self, num_channels, pool_features):
        super().__init__()
        self.branch1x1 = ConvNormActivation(
            in_channels=num_channels, out_channels=64, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )

        self.branch5x5_1 = ConvNormActivation(
            in_channels=num_channels, out_channels=48, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )
        self.branch5x5_2 = ConvNormActivation(
            in_channels=48, out_channels=64, kernel_size=5, padding=2, activation_layer=nn.ReLU
        )

        self.branch3x3dbl_1 = ConvNormActivation(
            in_channels=num_channels, out_channels=64, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )
        self.branch3x3dbl_2 = ConvNormActivation(
            in_channels=64, out_channels=96, kernel_size=3, padding=1, activation_layer=nn.ReLU
        )
        self.branch3x3dbl_3 = ConvNormActivation(
            in_channels=96, out_channels=96, kernel_size=3, padding=1, activation_layer=nn.ReLU
        )
        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        self.branch_pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=True)
        self.branch_pool_conv = ConvNormActivation(
            in_channels=num_channels, out_channels=pool_features, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        x = paddle.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)
        return x


class InceptionC(nn.Layer):
    def __init__(self, num_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = ConvNormActivation(
            in_channels=num_channels, out_channels=192, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )

        self.branch7x7_1 = ConvNormActivation(
            in_channels=num_channels,
            out_channels=channels_7x7,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=nn.ReLU,
        )
        self.branch7x7_2 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=channels_7x7,
            kernel_size=(1, 7),
            stride=1,
            padding=(0, 3),
            activation_layer=nn.ReLU,
        )
        self.branch7x7_3 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=192,
            kernel_size=(7, 1),
            stride=1,
            padding=(3, 0),
            activation_layer=nn.ReLU,
        )

        self.branch7x7dbl_1 = ConvNormActivation(
            in_channels=num_channels, out_channels=channels_7x7, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )
        self.branch7x7dbl_2 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=channels_7x7,
            kernel_size=(7, 1),
            padding=(3, 0),
            activation_layer=nn.ReLU,
        )
        self.branch7x7dbl_3 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=channels_7x7,
            kernel_size=(1, 7),
            padding=(0, 3),
            activation_layer=nn.ReLU,
        )
        self.branch7x7dbl_4 = ConvNormActivation(
            in_channels=channels_7x7,
            out_channels=channels_7x7,
            kernel_size=(7, 1),
            padding=(3, 0),
            activation_layer=nn.ReLU,
        )
        self.branch7x7dbl_5 = ConvNormActivation(
            in_channels=channels_7x7, out_channels=192, kernel_size=(1, 7), padding=(0, 3), activation_layer=nn.ReLU
        )
        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        self.branch_pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=True)
        self.branch_pool_conv = ConvNormActivation(
            in_channels=num_channels, out_channels=192, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        x = paddle.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)

        return x


class InceptionE_1(nn.Layer):
    def __init__(self, num_channels):
        super().__init__()
        self.branch1x1 = ConvNormActivation(
            in_channels=num_channels, out_channels=320, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )
        self.branch3x3_1 = ConvNormActivation(
            in_channels=num_channels, out_channels=384, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )
        self.branch3x3_2a = ConvNormActivation(
            in_channels=384, out_channels=384, kernel_size=(1, 3), padding=(0, 1), activation_layer=nn.ReLU
        )
        self.branch3x3_2b = ConvNormActivation(
            in_channels=384, out_channels=384, kernel_size=(3, 1), padding=(1, 0), activation_layer=nn.ReLU
        )

        self.branch3x3dbl_1 = ConvNormActivation(
            in_channels=num_channels, out_channels=448, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )
        self.branch3x3dbl_2 = ConvNormActivation(
            in_channels=448, out_channels=384, kernel_size=3, padding=1, activation_layer=nn.ReLU
        )
        self.branch3x3dbl_3a = ConvNormActivation(
            in_channels=384, out_channels=384, kernel_size=(1, 3), padding=(0, 1), activation_layer=nn.ReLU
        )
        self.branch3x3dbl_3b = ConvNormActivation(
            in_channels=384, out_channels=384, kernel_size=(3, 1), padding=(1, 0), activation_layer=nn.ReLU
        )

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        self.branch_pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=True)
        self.branch_pool_conv = ConvNormActivation(
            in_channels=num_channels, out_channels=192, kernel_size=1, padding=0, activation_layer=nn.ReLU
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = paddle.concat(branch3x3, axis=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = paddle.concat(branch3x3dbl, axis=1)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        x = paddle.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)
        return x


class InceptionE_2(InceptionE_1):
    def __init__(self, num_channels):
        super(InceptionE_2, self).__init__(num_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = paddle.concat(branch3x3, axis=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = paddle.concat(branch3x3dbl, axis=1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool_conv(branch_pool)

        x = paddle.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)
        return x

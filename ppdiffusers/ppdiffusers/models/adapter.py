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

from typing import List, Optional

import paddle

from ..configuration_utils import ConfigMixin, register_to_config
from .modeling_utils import ModelMixin
from .resnet import Downsample2D


class BottleneckResnetBlock(paddle.nn.Layer):
    def __init__(self, in_c, mid_c, out_c, down, ksize=3, sk=False, use_conv=True, proj_ksize=1):
        super().__init__()
        ps = ksize // 2
        proj_pad = proj_ksize // 2
        if in_c != mid_c or sk is False:
            self.conv1 = paddle.nn.Conv2D(
                in_channels=in_c, out_channels=mid_c, kernel_size=proj_ksize, stride=1, padding=proj_pad
            )
        else:
            self.conv1 = None
        if out_c != mid_c:
            self.conv2 = paddle.nn.Conv2D(
                in_channels=mid_c, out_channels=out_c, kernel_size=proj_ksize, stride=1, padding=proj_pad
            )
        else:
            self.conv2 = None
        self.block1 = paddle.nn.Conv2D(in_channels=mid_c, out_channels=mid_c, kernel_size=3, stride=1, padding=1)
        self.act = paddle.nn.ReLU()
        self.block2 = paddle.nn.Conv2D(in_channels=mid_c, out_channels=mid_c, kernel_size=ksize, stride=1, padding=ps)
        if sk is False:
            self.conv_shortcut = paddle.nn.Conv2D(
                in_channels=in_c, out_channels=mid_c, kernel_size=ksize, stride=1, padding=ps
            )
        else:
            self.conv_shortcut = None
        self.down = down
        if self.down is True:
            self.downsample = Downsample2D(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down is True:
            x = self.downsample(x)
        if self.conv1 is not None:
            x = self.conv1(x)
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.conv_shortcut is not None:
            h = h + self.conv_shortcut(x)
        else:
            h = h + x
        if self.conv2 is not None:
            h = self.conv2(h)
        return h


class T2IAdapter(ModelMixin, ConfigMixin):
    """
    A simple ResNet-like model that accepts images containing control signals such as keyposes and depth. The model
    generates multiple feature maps that are used as additional conditioning in [`UNet2DConditionModel`]. The model's
    architecture follows the original implementation of
    [Adapter](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L97)
     and
     [AdapterLight](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L235).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        block_out_channels (`List[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The number of channel of each downsample block's output hidden state. The `len(block_out_channels)` will
            also determine the number of downsample blocks in the Adapter.
        block_mid_channels (`List[int]`, *optional*, defaults to `block_out_channels` if not provided):
            The number of channels ResNet blocks in each downsample blocks will have, a downsample block will insert a
             projection layer in the last ResNet block when having different "mid_channel" and "out_channel".
        num_res_blocks (`int`, *optional*, defaults to 3):
            Number of ResNet blocks in each downsample block
        channels_in (`int`, *optional*, defaults to 3):
            Number of channels of Aapter's input(*control image*). Set this parameter to 1 if you're using gray scale
            image as *control image*.
        kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of conv-2d layers inside ResNet blocks.
        proj_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of conv-2d projection layers located at the start and end of a downsample block.
        res_block_skip (`bool`, *optional*, defaults to True):
            If set to `True`, ResNet block will using a regular residual connect that add layer's input to its output.
            If set to `False`, ResNet block will create a additional conv-2d layer in residual connect before adding
            residual back.
        use_conv (`bool`, *optional*, defaults to False):
            Whether to use a conv-2d layer for down sample feature map or a average pooling layer.
        input_scale_factor (`int`, *optional*, defaults to 8):
            The down scaling factor will be apply to input image when it is frist deliver to Adapter. Which should be
            equal to the down scaling factor of the VAE of your choice.
    """

    @register_to_config
    def __init__(
        self,
        block_out_channels: List[int] = [320, 640, 1280, 1280],
        block_mid_channels: Optional[List[int]] = None,
        num_res_blocks: int = 3,
        channels_in: int = 3,
        kernel_size: int = 3,
        proj_kernel_size: int = 1,
        res_block_skip: bool = True,
        use_conv: bool = False,
        input_scale_factor: int = 8,
    ):
        super(T2IAdapter, self).__init__()
        self.num_downsample_blocks = len(block_out_channels)
        self.unshuffle = paddle.nn.PixelUnshuffle(downscale_factor=input_scale_factor)
        self.num_res_blocks = num_res_blocks
        self.body = []
        if block_mid_channels is None:
            block_mid_channels = block_out_channels
        for i in range(self.num_downsample_blocks):
            for j in range(num_res_blocks):
                if i != 0 and j == 0:
                    self.body.append(
                        BottleneckResnetBlock(
                            block_out_channels[i - 1],
                            block_mid_channels[i],
                            block_mid_channels[i],
                            down=True,
                            ksize=kernel_size,
                            proj_ksize=proj_kernel_size,
                            sk=res_block_skip,
                            use_conv=use_conv,
                        )
                    )
                elif j == num_res_blocks - 1:
                    self.body.append(
                        BottleneckResnetBlock(
                            block_mid_channels[i],
                            block_mid_channels[i],
                            block_out_channels[i],
                            down=False,
                            ksize=kernel_size,
                            proj_ksize=proj_kernel_size,
                            sk=res_block_skip,
                            use_conv=use_conv,
                        )
                    )
                else:
                    self.body.append(
                        BottleneckResnetBlock(
                            block_mid_channels[i],
                            block_mid_channels[i],
                            block_mid_channels[i],
                            down=False,
                            ksize=kernel_size,
                            proj_ksize=proj_kernel_size,
                            sk=res_block_skip,
                            use_conv=use_conv,
                        )
                    )
        self.body = paddle.nn.LayerList(sublayers=self.body)
        if block_mid_channels[0] == block_out_channels[0]:
            self.conv_in = paddle.nn.Conv2D(
                in_channels=channels_in * input_scale_factor**2,
                out_channels=block_mid_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv_in = paddle.nn.Conv2D(
                in_channels=channels_in * input_scale_factor**2,
                out_channels=block_mid_channels[0],
                kernel_size=proj_kernel_size,
                stride=1,
                padding=proj_kernel_size // 2,
            )

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                (batch, channel, height, width) input images for adapter model, `channel` should equal to
                `channels_in`.
        """
        x = self.unshuffle(x)
        features = []
        x = self.conv_in(x)
        for i in range(self.num_downsample_blocks):
            for j in range(self.num_res_blocks):
                idx = i * self.num_res_blocks + j
                x = self.body[idx](x)
            features.append(x)
        return features


class MultiAdapter(ModelMixin):
    """
    MultiAdapter is a wrapper model that contains multiple adapter models and merges their outputs according to
    user-assigned weighting.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        adapters (`List[T2IAdapter]`, *optional*, defaults to None):
            A list of `T2IAdapter` model instances.
    """

    def __init__(self, adapters: List[T2IAdapter]):
        super(MultiAdapter, self).__init__()
        self.num_adapter = len(adapters)
        self.adapters = paddle.nn.LayerList(sublayers=adapters)

    def forward(self, xs: paddle.Tensor, adapter_weights: Optional[List[float]] = None) -> List[paddle.Tensor]:
        """
        Args:
            xs (`torch.Tensor`):
                (batch, channel, height, width) input images for multiple adapter models concated along dimension 1,
                `channel` should equal to `num_adapter` * "number of channel of image".
            adapter_weights (`List[float]`, *optional*, defaults to None):
                List of floats representing the weight which will be multiply to each adapter's output before adding
                them together.
        """
        if adapter_weights is None:
            adapter_weights = paddle.to_tensor([1 / self.num_adapter] * self.num_adapter)
        else:
            adapter_weights = paddle.to_tensor(adapter_weights)
        if xs.shape[1] % self.num_adapter != 0:
            raise ValueError(
                f"Expecting multi-adapter's input have number of channel that cab be evenly divisible by num_adapter: {xs.shape[1]} % {self.num_adapter} != 0"
            )
        x_list = paddle.chunk(x=xs, chunks=self.num_adapter, axis=1)
        accume_state = None
        for x, w, adapter in zip(x_list, adapter_weights, self.adapters):
            features = adapter(x)
            if accume_state is None:
                accume_state = features
            else:
                for i in range(len(features)):
                    accume_state[i] += w * features[i]
        return accume_state

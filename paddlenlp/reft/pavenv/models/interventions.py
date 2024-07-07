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

from abc import abstractmethod

import paddle
from paddle import nn

from .basic_utils import sigmoid_boundary
from .intervention_utils import _do_intervention_by_swap
from .layers import RotateLayer


class Intervention(nn.Layer):
    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.trainable = False
        self.is_source_constant = False

        self.keep_last_dim = kwargs["keep_last_dim"] if "keep_last_dim" in kwargs else False
        self.use_fast = kwargs["use_fast"] if "use_fast" in kwargs else False
        self.subspace_partition = kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        # we turn the partition into list indices
        if self.subspace_partition is not None:
            expanded_subspace_partition = []
            for subspace in self.subspace_partition:
                if len(subspace) == 2 and isinstance(subspace[0], int):
                    expanded_subspace_partition.append([i for i in range(subspace[0], subspace[1])])
                else:
                    # it could be discrete indices.
                    expanded_subspace_partition.append(subspace)
            self.subspace_partition = expanded_subspace_partition

        if "embed_dim" in kwargs and kwargs["embed_dim"] is not None:
            self.register_buffer("embed_dim", paddle.to_tensor(kwargs["embed_dim"]))
            self.register_buffer("interchange_dim", paddle.to_tensor(kwargs["embed_dim"]))
        else:
            self.embed_dim = None
            self.interchange_dim = None

        if "source_representation" in kwargs and kwargs["source_representation"] is not None:
            self.is_source_constant = True
            self.register_buffer("source_representation", kwargs["source_representation"])
        else:
            if "hidden_source_representation" in kwargs and kwargs["hidden_source_representation"] is not None:
                self.is_source_constant = True
            else:
                self.source_representation = None

    def set_source_representation(self, source_representation):
        self.is_source_constant = True
        self.register_buffer("source_representation", source_representation)

    def set_interchange_dim(self, interchange_dim):
        if isinstance(interchange_dim, int):
            self.interchange_dim = paddle.to_tensor(interchange_dim)
        else:
            self.interchange_dim = interchange_dim

    @abstractmethod
    def forward(self, base, source, subspaces=None):
        pass


class LocalistRepresentationIntervention(nn.Layer):
    """Localist representation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.is_repr_distributed = False


class VanillaIntervention(Intervention, LocalistRepresentationIntervention):
    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, base, source, subspaces=None):
        return _do_intervention_by_swap(
            base,
            (source if self.source_representation is None else self.source_representation),
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return f"VanillaIntervention(){1}"


class TrainableIntervention(Intervention):
    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = True
        self.is_source_constant = False

    def tie_weight(self, linked_intervention):
        pass


class BasisAgnosticIntervention(Intervention):
    """Intervention that will modify its basis in a uncontrolled manner."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.basis_agnostic = True


class SkipIntervention(BasisAgnosticIntervention, LocalistRepresentationIntervention):
    """Skip the current intervening layer's computation in the hook function."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, base, source, subspaces=None):
        # source here is the base example input to the hook
        return _do_intervention_by_swap(
            base,
            source,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return "SkipIntervention()"


class ConstantSourceIntervention(Intervention):
    """Constant source."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_source_constant = True


class CollectIntervention(ConstantSourceIntervention):
    """Collect activations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, base, source=None, subspaces=None):
        return _do_intervention_by_swap(
            base,
            source,
            "collect",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return "CollectIntervention()"


class DistributedRepresentationIntervention(nn.Layer):
    """Distributed representation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.is_repr_distributed = True


class BoundlessRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):
    """Intervention in the rotated space with boundary mask."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = nn.initializer.Orthogonal()(rotate_layer.weight)
        self.intervention_boundaries = self.create_parameter(
            shape=[1],
            default_initializer=nn.initializer.Assign(paddle.to_tensor([0.5])),
            is_bias=False,
        )
        self.temperature = self.create_parameter(
            shape=[1],
            default_initializer=nn.initializer.Assign(paddle.to_tensor(50.0)),
            is_bias=False,
        )
        self.intervention_population = self.create_parameter(
            shape=[self.embed_dim],
            default_initializer=nn.initializer.Assign(paddle.arange(0, self.embed_dim)),
            is_bias=False,
            stop_gradient=True,
        )

    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: paddle.Tensor):
        self.temperature.set_value(temp)

    def set_intervention_boundaries(self, intervention_boundaries):
        self.intervention_boundaries.set_value(paddle.to_tensor([intervention_boundaries]))

    def forward(self, base, source, subspaces=None):
        batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = paddle.clip(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.tile([batch_size, 1]),
            0.0,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature,
        )
        boundary_mask = paddle.ones([batch_size, 1], dtype=base.dtype) * boundary_mask
        boundary_mask = boundary_mask.astype(rotated_base.dtype)
        # interchange
        rotated_output = (1.0 - boundary_mask) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = paddle.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.astype(base.dtype)

    def __str__(self):
        return "BoundlessRotatedSpaceIntervention()"


class SourcelessIntervention(Intervention):
    """No source."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_source_constant = True

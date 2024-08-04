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

import json
from collections import namedtuple

from paddlenlp.transformers.configuration_utils import PretrainedConfig

RepresentationConfig = namedtuple(
    "RepresentationConfig",
    "layer component unit "
    "max_number_of_units "
    "low_rank_dimension intervention_type intervention "
    "subspace_partition group_key intervention_link_key moe_key "
    "source_representation hidden_source_representation",
    defaults=(0, "block_output", "pos", 1, None, None, None, None, None, None, None, None, None),
)


class IntervenableConfig(PretrainedConfig):
    def __init__(
        self,
        representations=[RepresentationConfig()],
        intervention_types=None,
        sorted_keys=None,
        model_type=None,  # deprecating
        intervention_dimensions=None,
        intervention_constant_sources=None,
        **kwargs,
    ):
        if not isinstance(representations, list):
            representations = [representations]

        casted_representations = []
        for reprs in representations:
            print(f"type is : {type(reprs)}")
            if isinstance(reprs, RepresentationConfig):
                casted_representations += [reprs]
            elif isinstance(reprs, list):
                casted_representations += [RepresentationConfig(*reprs)]
            elif isinstance(reprs, dict):
                casted_representations += [RepresentationConfig(**reprs)]
            else:
                raise ValueError(f"{reprs} format in our representation list is not supported.")
        self.representations = casted_representations
        self.intervention_types = intervention_types
        # the type inside reprs can overwrite
        overwrite = False
        overwrite_intervention_types = []
        for reprs in self.representations:
            if overwrite:
                if reprs.intervention_type is None and reprs.intervention is None:
                    raise ValueError("intervention_type if used should be specified for all")
            if reprs.intervention_type is not None:
                overwrite = True
                overwrite_intervention_types += [reprs.intervention_type]
            elif reprs.intervention is not None:
                overwrite = True
                overwrite_intervention_types += [type(reprs.intervention)]
            if reprs.intervention_type is not None and reprs.intervention is not None:
                raise ValueError("Only one of the field should be provided: intervention_type, intervention")

        if overwrite:
            self.intervention_types = overwrite_intervention_types

        self.sorted_keys = sorted_keys
        self.intervention_dimensions = intervention_dimensions
        self.intervention_constant_sources = intervention_constant_sources
        self.model_type = model_type
        super().__init__(**kwargs)

    def __repr__(self):
        representations = []
        for reprs in self.representations:
            if isinstance(reprs, list):
                reprs = RepresentationConfig(*reprs)
            new_d = {}
            for k, v in reprs._asdict().items():
                if type(v) not in {str, int, list, tuple, dict} and v is not None and v != [None]:
                    new_d[k] = "PLACEHOLDER"
                else:
                    new_d[k] = v
            representations += [new_d]
        _repr = {
            "model_type": str(self.model_type),
            "representations": tuple(representations),
            "intervention_types": str(self.intervention_types),
            "sorted_keys": tuple(self.sorted_keys) if self.sorted_keys is not None else str(self.sorted_keys),
            "intervention_dimensions": str(self.intervention_dimensions),
        }
        _repr_string = json.dumps(_repr, indent=4)
        return f"IntervenableConfig\n{_repr_string}"

    def __str__(self):
        return self.__repr__()

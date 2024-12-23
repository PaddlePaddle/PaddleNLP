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

from .causal_conv1d_varlen import (
    causal_conv1d_varlen_states,
    causal_conv1d_varlen_states_ref,
)
from .k_activations import swiglu
from .selective_state_update import selective_state_update, selective_state_update_ref
from .ssd_chunk_scan import chunk_scan, chunk_scan_ref
from .ssd_chunk_state import chunk_state, chunk_state_ref
from .ssd_combined import (
    mamba_chunk_scan,
    mamba_chunk_scan_combined,
    mamba_conv1d_scan_ref,
    mamba_split_conv1d_scan_combined,
    mamba_split_conv1d_scan_ref,
    ssd_selective_scan,
)
from .ssd_state_passing import state_passing, state_passing_ref

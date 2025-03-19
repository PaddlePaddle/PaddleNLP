// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "xpu/kernel/cluster.h"

template <int LOAD_CORE>
struct XGroup {
    int group_num{0};
    int CID{0};
    int soft_group_id{0};
    int soft_core_id{0};
    unsigned int hard_grp_sync_int{0};

    __device__ XGroup() {
        group_num = core_num() / LOAD_CORE;
        const int hard_grp_id = core_id() % 16;
        const int in_hard_grp_id = core_id() / 16;
        CID = hard_grp_id * 4 + in_hard_grp_id;
        soft_group_id = CID / LOAD_CORE;
        soft_core_id = CID % LOAD_CORE;
        const int in_grp_hard_grp = max(1, LOAD_CORE / 4);
        const int start_hard_grp =
                (LOAD_CORE >= 4) ? soft_group_id * in_grp_hard_grp : soft_group_id / 2;
        hard_grp_sync_int = ((1 << (start_hard_grp + in_grp_hard_grp)) - (1 << start_hard_grp));
    }

    inline __device__ bool is_last_core_in_group() {
        return soft_core_id == LOAD_CORE - 1;
    }

    inline __device__ bool is_first_core_in_group() {
        return soft_core_id == 0;
    }

    template <typename T>
    inline __device__ void reduce_max(T& max_value, __shared_ptr__ T* sm_tmp_) {
#if LOAD_CORE > 1
        sm_tmp_[CID] = max_value;
        mfence_sm();
        sync_group(hard_grp_sync_int);
        for (int i = 0; i < LOAD_CORE; i++) {
            max_value = fmax(max_value, sm_tmp_[soft_group_id * LOAD_CORE + i]);
        }
#endif
    }

    template <typename T>
    inline __device__ void reduce_sum(T& sum_value, __shared_ptr__ T* sm_tmp_) {
#if LOAD_CORE > 1
        sm_tmp_[CID] = sum_value;
        mfence_sm();
        sync_group(hard_grp_sync_int);
        sum_value = 0.0f;
        // optimize this with group_sum
        for (int i = 0; i < LOAD_CORE; i++) {
            sum_value = sum_value + sm_tmp_[soft_group_id * LOAD_CORE + i];
        }
#endif
    }
};
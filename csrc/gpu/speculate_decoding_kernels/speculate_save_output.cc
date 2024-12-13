// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include "paddle/extension.h"

#define SPECULATE_MAX_BSZ 256
#define MAX_DRAFT_TOKENS 6

struct msgdata {
    long mtype;
    int mtext[SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2];   // stop_flag, bsz, accept_num*bsz, tokens...
};

void SpeculateSaveWithOutputMsg(const paddle::Tensor& accept_tokens,
                 const paddle::Tensor& accept_num,
                 const paddle::Tensor& not_need_stop,
                 int64_t rank_id) {            
    if (rank_id > 0) return;

    int max_draft_tokens = accept_tokens.shape()[1];

    auto accept_tokens_cpu = accept_tokens.copy_to(paddle::CPUPlace(), false);
    auto accept_num_cpu = accept_num.copy_to(paddle::CPUPlace(), false);
    auto not_need_stop_cpu = not_need_stop.copy_to(paddle::CPUPlace(), false);

    int64_t *accept_tokens_data = accept_tokens_cpu.data<int64_t>();
    int *accept_num_data = accept_num_cpu.data<int>();
    bool not_need_stop_data = not_need_stop_cpu.data<bool>()[0];

    static struct msgdata msg_sed;
    static key_t key = ftok("./", 1);
    static int msgid = msgget(key, IPC_CREAT | 0666);

    msg_sed.mtype = 1;
    msg_sed.mtext[0] = not_need_stop_data ? 1 : -1;
    int bsz = accept_tokens.shape()[0];
    msg_sed.mtext[1] = bsz;

    for (int i = 2; i < SPECULATE_MAX_BSZ + 2; i++) {
        if (i - 2 >= bsz) {
            msg_sed.mtext[i] = 0;
        } else {
            msg_sed.mtext[i] = (int)accept_num_data[i - 2];
        }
    }
    for (int i = SPECULATE_MAX_BSZ + 2; i < SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2; i++) {
        int token_id = i - SPECULATE_MAX_BSZ - 2;
        int bid = token_id / MAX_DRAFT_TOKENS;
        int local_token_id = token_id % MAX_DRAFT_TOKENS;
        if (token_id / MAX_DRAFT_TOKENS >= bsz) {
            msg_sed.mtext[i] = 0;
        } else {
            msg_sed.mtext[i] = accept_tokens_data[bid * max_draft_tokens + local_token_id];
        }
    }
    if ((msgsnd(msgid, &msg_sed, (SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2) * 4, 0)) == -1) {
      printf("full msg buffer\n");
    }
    return;
}

PD_BUILD_OP(speculate_save_output)
    .Inputs({"accept_tokens", "accept_num", "not_need_stop"})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"accept_tokens", "x_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateSaveWithOutputMsg));

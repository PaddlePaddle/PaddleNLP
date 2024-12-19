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

void SpeculateGetOutput(const paddle::Tensor& x,
        int64_t rank_id,
        bool wait_flag) {
    if (rank_id > 0) {
        return;
    }
    static struct msgdata msg_rcv;

    static key_t key = ftok("./", 1);

    static int msgid = msgget(key, IPC_CREAT | 0666);

    int64_t *out_data = const_cast<int64_t*>(x.data<int64_t>());
    int ret = -1;
    if (!wait_flag) {
        ret = msgrcv(msgid, &msg_rcv, (SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2) * 4, 0, IPC_NOWAIT);
    } else {
        ret = msgrcv(msgid, &msg_rcv, (SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2) * 4, 0, 0);
    }
    if(ret == -1) {
        out_data[0] = -2;
        out_data[1] = 0;
        return;
    }
    int bsz = msg_rcv.mtext[1];

    for (int64_t i = 0; i < SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2; i++) {
        out_data[i] = (int64_t)msg_rcv.mtext[i];
    }
    return;
}

PD_BUILD_OP(speculate_get_output)
    .Inputs({"x"})
    .Attrs({"rank_id: int64_t",
            "wait_flag: bool"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateGetOutput));
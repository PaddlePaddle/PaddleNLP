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

#define MAX_BSZ 512
// #define SAVE_WITH_OUTPUT_DEBUG

struct msgdata {
    long mtype;
    int mtext[MAX_BSZ + 2];   // stop_flag, bsz, tokens
};

void SaveOutMmsg(const paddle::Tensor& x,
                 const paddle::Tensor& not_need_stop, // cpu
                 int64_t rank_id) {
    if (rank_id > 0) return;
    auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
    int64_t *x_data = x_cpu.data<int64_t>();
    auto not_need_stop_data = not_need_stop.data<bool>()[0];

    static struct msgdata msg_sed;
    int msg_queue_id = 1;
    if (const char* inference_msg_queue_id_env_p =
            std::getenv("INFERENCE_MSG_QUEUE_ID")) {
        std::string inference_msg_queue_id_env_str(
            inference_msg_queue_id_env_p);
        int inference_msg_queue_id_from_env =
            std::stoi(inference_msg_queue_id_env_str);
        msg_queue_id = inference_msg_queue_id_from_env;
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "Your INFERENCE_MSG_QUEUE_ID is: "
                  << inference_msg_queue_id_from_env << std::endl;
#endif
    } else {
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "Failed to got INFERENCE_MSG_QUEUE_ID at env, use default."
                  << std::endl;
#endif
    }
    static key_t key = ftok("./", msg_queue_id);
    static int msgid = msgget(key, IPC_CREAT | 0666);

    msg_sed.mtype = 1;
    msg_sed.mtext[0] = not_need_stop_data ? 1 : -1;
    int bsz = x.shape()[0];
    msg_sed.mtext[1] = bsz;
    for (int i = 2; i < bsz + 2; i++) {
        msg_sed.mtext[i] = (int)x_data[i - 2];
    }
    if ((msgsnd(msgid, &msg_sed, (MAX_BSZ + 2) * 4, 0)) == -1) {
    //   printf("full msg buffer\n");
    }
    return;
}

PD_BUILD_OP(save_output)
    .Inputs({"x", "not_need_stop"})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutMmsg));
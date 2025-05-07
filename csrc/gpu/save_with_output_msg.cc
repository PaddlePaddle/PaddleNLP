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

#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include "paddle/extension.h"

#define MAX_BSZ 256

// #define SAVE_WITH_OUTPUT_DEBUG
struct msgdata {
    long mtype;
    int mtext[MAX_BSZ + 2];   // stop_flag, bsz, tokens
};

void SaveOutMmsg(const paddle::Tensor& x,
                 const paddle::Tensor& not_need_stop,
                 int64_t rank_id,
                 int msg_queue_id) {

    if (rank_id > 0) return;
    auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
    int64_t *x_data = x_cpu.data<int64_t>();
    static struct msgdata msg_sed;

    if (const char* inference_msg_queue_id_env_p = std::getenv("INFERENCE_MSG_QUEUE_ID")){
        std::string inference_msg_queue_id_env_str(inference_msg_queue_id_env_p);
        int inference_msg_queue_id_from_env = std::stoi(inference_msg_queue_id_env_str);
        msg_queue_id = inference_msg_queue_id_from_env;
#ifdef SAVE_WITH_OUTPUT_DEBUG
            std::cout << "Your INFERENCE_MSG_QUEUE_ID is: " << inference_msg_queue_id_from_env << std::endl;
#endif
    } else {
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "Failed to got INFERENCE_MSG_QUEUE_ID at env, use default." << std::endl;
#endif
    }
    int inference_msg_id_from_env = 1;
    if (const char* inference_msg_id_env_p = std::getenv("INFERENCE_MSG_ID")){
        std::string inference_msg_id_env_str(inference_msg_id_env_p);
        inference_msg_id_from_env = std::stoi(inference_msg_id_env_str);
        if (inference_msg_id_from_env == 2){
            // 2 and -2 is perserve for no-output indication.
            throw std::runtime_error(" INFERENCE_MSG_ID cannot be 2, please use other number.");
        }
        if (inference_msg_id_from_env < 0) {
            throw std::runtime_error(" INFERENCE_MSG_ID cannot be negative, please use other number.");
        }

#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "Your INFERENCE_MSG_ID is: " << inference_msg_id_from_env << std::endl;
#endif
    } else {
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "Failed to got INFERENCE_MSG_ID at env, use (int)1 as default." << std::endl;
#endif
    }
    static key_t key = ftok("/dev/shm", msg_queue_id);

    static int msgid = msgget(key, IPC_CREAT | 0666);
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "save_output_key: " << key << std::endl;
        std::cout << "save msgid: " << msgid << std::endl;
#endif  
    msg_sed.mtype = 1;
    bool not_need_stop_data = not_need_stop.data<bool>()[0];
    // printf("not_need_stop_data %d\n", (int)not_need_stop_data);
    msg_sed.mtext[0] = not_need_stop_data ? inference_msg_id_from_env : -inference_msg_id_from_env;
    int bsz = x.shape()[0];
    msg_sed.mtext[1] = bsz;
    for (int i = 2; i < bsz + 2; i++) {
        msg_sed.mtext[i] = (int)x_data[i - 2];
    }
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "msg data: ";
        for (int i = 0; i < bsz; i++) {
            std::cout << " " << (int)x_data[i];
        }
        std::cout << std::endl;
#endif
    if ((msgsnd(msgid, &msg_sed, (MAX_BSZ + 2) * 4, 0)) == -1) {
      printf("full msg buffer\n");
    }
    return;
}


void SaveOutMmsgStatic(const paddle::Tensor& x,
                 const paddle::Tensor& not_need_stop,
                 int64_t rank_id) {
    SaveOutMmsg(x, not_need_stop, rank_id, 1);
}

void SaveOutMmsgDynamic(const paddle::Tensor& x,
                 const paddle::Tensor& not_need_stop,
                 int64_t rank_id,
                 int msg_queue_id) {
    SaveOutMmsg(x, not_need_stop, rank_id, msg_queue_id);
}

PD_BUILD_OP(save_output)
    .Inputs({"x", "not_need_stop"})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutMmsgStatic));

PD_BUILD_OP(save_output_dynamic)
    .Inputs({"x", "not_need_stop"})
    .Attrs({"rank_id: int64_t", "msg_queue_id: int"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutMmsgDynamic));
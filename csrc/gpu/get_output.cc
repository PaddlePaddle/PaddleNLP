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
// #define GET_OUTPUT_DEBUG
struct msgdata {
    long mtype;
    int mtext[MAX_BSZ + 2];   // stop_flag, bsz, tokens
};

void GetOutput(const paddle::Tensor& x,
        int64_t rank_id,
        bool wait_flag,
        int msg_queue_id) {
    if (rank_id > 0) {
        return;
    }
    static struct msgdata msg_rcv;
    if (const char* inference_msg_queue_id_env_p = std::getenv("INFERENCE_MSG_QUEUE_ID")){
        std::string inference_msg_queue_id_env_str(inference_msg_queue_id_env_p);
        int inference_msg_queue_id_from_env = std::stoi(inference_msg_queue_id_env_str);
#ifdef GET_OUTPUT_DEBUG
            std::cout << "Your INFERENCE_MSG_QUEUE_ID is: " << inference_msg_queue_id_from_env << std::endl;
#endif
        msg_queue_id = inference_msg_queue_id_from_env;
    }
    static key_t key = ftok("/dev/shm", msg_queue_id);
    static int msgid = msgget(key, IPC_CREAT | 0666);

#ifdef GET_OUTPUT_DEBUG
        std::cout<<"get_output_key: "<< key << std::endl;
        std::cout << "get_output msgid: " << msgid << std::endl;
#endif  

    int64_t *out_data = const_cast<int64_t*>(x.data<int64_t>());
    int ret = -1;
    if (!wait_flag) {
        ret = msgrcv(msgid, &msg_rcv, (MAX_BSZ + 2) * 4, 0, IPC_NOWAIT);
    } else {
        ret = msgrcv(msgid, &msg_rcv, (MAX_BSZ + 2) * 4, 0, 0);
    }
    if(ret == -1) {
        out_data[0] = -2;
        out_data[1] = 0;
        return;
    }
    int bsz = msg_rcv.mtext[1];

    for (int64_t i = 0; i < bsz + 2; i++) {
        out_data[i] = (int64_t)msg_rcv.mtext[i];
    }
#ifdef GET_OUTPUT_DEBUG
        std::cout << "get_output finished: " << msgid << std::endl;
#endif  

    return;
}

void GetOutputStatic(const paddle::Tensor& x,
        int64_t rank_id,
        bool wait_flag) {
    GetOutput(x, rank_id, wait_flag, 1);
}

void GetOutputDynamic(const paddle::Tensor& x,
        int64_t rank_id,
        bool wait_flag,
        int msg_queue_id) {
    
    GetOutput(x, rank_id, wait_flag, msg_queue_id);
}

PD_BUILD_OP(get_output)
    .Inputs({"x"})
    .Attrs({"rank_id: int64_t",
            "wait_flag: bool"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(GetOutputStatic));

PD_BUILD_OP(get_output_dynamic)
    .Inputs({"x"})
    .Attrs({"rank_id: int64_t",
            "wait_flag: bool",
            "msg_queue_id: int"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(GetOutputDynamic));
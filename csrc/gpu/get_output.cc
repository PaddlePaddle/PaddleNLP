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

#define MAX_BSZ 256
#define MAX_DRAFT_TOKENS 6

template <int SIZE>
struct MsgData {
  long mtype;
  std::array<int, SIZE> mtext;
};

template <int SIZE>
void GetOutputFunc(MsgData<SIZE>& msg_rcv,  // NOLINT
                   const paddle::Tensor& x,
                   int64_t rank_id,
                   bool wait_flag) {
  if (rank_id > 0) return;

  static key_t key = ftok("./", 1);

  static int msgid = msgget(key, IPC_CREAT | 0666);

  int ret = -1;
  ret = msgrcv(
      msgid, &msg_rcv, SIZE * sizeof(int), 0, wait_flag ? 0 : IPC_NOWAIT);

  int64_t* out_data = const_cast<int64_t*>(x.data<int64_t>());

  if (ret == -1) {
    // read none
    out_data[0] = -2;
    out_data[1] = 0;
    return;
  }

  for (int64_t i = 0; i < SIZE; i++) {
    out_data[i] = (int64_t)msg_rcv.mtext[i];
  }

  return;
}

void GetOutput(const paddle::Tensor& x,
               int64_t rank_id,
               bool wait_flag,
               bool speculative_decoding) {
  if (!speculative_decoding) {
    constexpr int SIZE = MAX_BSZ + 2;  // stop_flag, bsz, tokens...
    static struct MsgData<SIZE> msg_rcv;
    GetOutputFunc<SIZE>(msg_rcv, x, rank_id, wait_flag);
  } else {
    constexpr int SIZE = MAX_BSZ * MAX_DRAFT_TOKENS +
                         MAX_BSZ +
                         2;  // stop_flag, bsz, accept_num*bsz, tokens...
    static struct MsgData<SIZE> specu_msg_rcv;
    GetOutputFunc<SIZE>(specu_msg_rcv, x, rank_id, wait_flag);
  }
}

PD_BUILD_OP(get_output)
    .Inputs({"x"})
    .Attrs({"rank_id: int64_t",
            "wait_flag: bool",
            "speculative_decoding: bool"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(GetOutput));
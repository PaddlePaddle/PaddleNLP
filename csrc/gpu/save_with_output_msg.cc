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
void SaveOutMsgFunc(MsgData<SIZE>& msg_sed,  // NOLINT
                    const paddle::Tensor& x,
                    const paddle::Tensor& not_need_stop,
                    const paddle::optional<paddle::Tensor>& accept_num,
                    int64_t rank_id) {
  if (rank_id > 0) return;
  auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
  int64_t* x_data = x_cpu.data<int64_t>();
  auto not_need_stop_cpu = not_need_stop.copy_to(paddle::CPUPlace(), false);
  bool* not_need_stop_data = not_need_stop_cpu.data<bool>();

  static key_t key = ftok("./", 1);
  static int msgid = msgget(key, IPC_CREAT | 0666);
  int bsz = x.shape()[0];

  if (!accept_num) {
    msg_sed.mtype = 1;
    msg_sed.mtext[0] = not_need_stop_data[0] ? 1 : -1;
    msg_sed.mtext[1] = bsz;
    for (int i = 2; i < bsz + 2; i++) {
      msg_sed.mtext[i] = (int)x_data[i - 2];
    }
    if ((msgsnd(msgid, &msg_sed, SIZE * sizeof(int), 0)) == -1) {
      //   printf("full msg buffer\n");
    }
  } else {
    auto accept_num_cpu = accept_num.get().copy_to(paddle::CPUPlace(), false);
    int* accept_num_data = accept_num_cpu.data<int>();

    msg_sed.mtype = 1;
    msg_sed.mtext[0] = not_need_stop_data[0] ? 1 : -1;
    msg_sed.mtext[1] = bsz;
    for (int i = 2; i < MAX_BSZ + 2; i++) {
      if (i - 2 >= bsz) {
        msg_sed.mtext[i] = 0;
      } else {
        msg_sed.mtext[i] = (int)accept_num_data[i - 2];
      }
    }
    for (int i = MAX_BSZ + 2; i < SIZE; i++) {
      int token_id = i - MAX_BSZ - 2;
      int bid = token_id / MAX_DRAFT_TOKENS;
      int local_token_id = token_id % MAX_DRAFT_TOKENS;
      if (token_id / MAX_DRAFT_TOKENS >= bsz) {
        msg_sed.mtext[i] = 0;
      } else {
        msg_sed.mtext[i] = x_data[bid * MAX_DRAFT_TOKENS + local_token_id];
      }
    }
    if ((msgsnd(msgid, &msg_sed, SIZE * sizeof(int), 0)) == -1) {
      printf("full msg buffer\n");
    }
  }

  return;
}

void SaveOutMsg(const paddle::Tensor& x,
                const paddle::Tensor& not_need_stop,
                const paddle::optional<paddle::Tensor>& accept_num,
                int64_t rank_id) {
  if (!accept_num) {
    constexpr int SIZE = MAX_BSZ + 2;  // stop_flag, bsz, tokens...
    static struct MsgData<SIZE> msg_sed;
    SaveOutMsgFunc<SIZE>(msg_sed, x, not_need_stop, accept_num, rank_id);
  } else {
    constexpr int SIZE = MAX_BSZ * MAX_DRAFT_TOKENS +
                         MAX_BSZ +
                         2;  // stop_flag, bsz, accept_num*bsz, tokens...
    static struct MsgData<SIZE> specu_msg_sed;
    SaveOutMsgFunc<SIZE>(specu_msg_sed, x, not_need_stop, accept_num, rank_id);
  }
}

PD_BUILD_OP(save_output)
    .Inputs({"x", "not_need_stop", paddle::Optional("accept_num")})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutMsg));
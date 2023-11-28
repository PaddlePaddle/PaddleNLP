#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include "paddle/extension.h"

#define MAX_BSZ 512

struct msgdata {
    long mtype;
    int mtext[MAX_BSZ + 2];   // stop_flag, bsz, tokens
};

void GetOutput(const paddle::Tensor& x,
               int64_t rank_id,
               bool wait_flag) {
  if (rank_id > 0) return;

  static struct msgdata msg_rcv;

  static key_t key = ftok("./", 1);

  static int msgid = msgget(key, IPC_CREAT | 0666);

  int64_t *out_data = const_cast<int64_t*>(x.data<int64_t>());
  int ret = -1;
  if (!wait_flag) {
    ret = msgrcv(msgid, &msg_rcv, (MAX_BSZ + 2) * 4, 0, IPC_NOWAIT);
  } else {
    ret = msgrcv(msgid, &msg_rcv, (MAX_BSZ + 2) * 4, 0, 0);
  }
  if(ret == -1)
	{
    // read none
    out_data[0] = -2;
    out_data[1] = 0;
		return;
	}

  int bsz = msg_rcv.mtext[1];

  for (int64_t i = 0; i < bsz + 2; i++) {
    out_data[i] = (int64_t)msg_rcv.mtext[i];
  }
  return;
}

PD_BUILD_OP(get_output)
    .Inputs({"x"})
    .Attrs({"rank_id: int64_t",
            "wait_flag: bool"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(GetOutput));

/*************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL1_COMPAT_H
#define NCCL1_COMPAT_H

#ifndef NCCL_MAJOR // NCCL 1.x
#define NCCL_MAJOR 1
#define NCCL_MINOR 0

#define ncclNumOps nccl_NUM_OPS
#define ncclNumTypes nccl_NUM_TYPES

static ncclResult_t ncclGroupStart() { return ncclSuccess; }
static ncclResult_t ncclGroupEnd() { return ncclSuccess; }

#define CHECKCOUNT(count) if (count > INT_MAX) return ncclInvalidArgument;

static ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  CHECKCOUNT(count);
  return ncclReduce(sendbuff, recvbuff, (int)count, datatype, op, root, comm, stream);
}
static ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  CHECKCOUNT(count);
  return ncclAllReduce(sendbuff, recvbuff, (int)count, datatype, op, comm, stream);
}
static ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  CHECKCOUNT(count);
  return ncclBcast(buff, (int)count, datatype, root, comm, stream);
}
static ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream) {
  CHECKCOUNT(recvcount);
  return ncclReduceScatter(sendbuff, recvbuff, (int)recvcount, datatype, op, comm, stream);
}
static ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  CHECKCOUNT(sendcount);
  return ncclAllGather(sendbuff, (int)sendcount, datatype, recvbuff, comm, stream);
}
#endif

#endif

/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void ScatterGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = (count/nranks)*nranks;
  *recvcount = count/nranks;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = count/nranks;
  *paramcount = count/nranks;
}

testResult_t ScatterInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    if (rank == root) TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, rep, 1, 0));
    TESTCHECK(InitData(args->expected[i], recvcount, rank*recvcount, type, ncclSum, rep, 1, 0));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

void ScatterGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t ScatterRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  size_t rankOffset = count * wordSize(type);
  if (count == 0) return testSuccess;

  NCCLCHECK(ncclGroupStart());
  if (rank == root) {
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, type, r, comm, stream));
    }
  }
  NCCLCHECK(ncclRecv(recvbuff, count, type, root, comm, stream));
  NCCLCHECK(ncclGroupEnd());

  return testSuccess;
}

struct testColl scatterTest = {
  "Scatter",
  ScatterGetCollByteCount,
  ScatterInitData,
  ScatterGetBw,
  ScatterRunColl
};

void ScatterGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ScatterGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t ScatterRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &scatterTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;
  int begin_root, end_root;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if (root != -1) {
    begin_root = end_root = root;
  } else {
    begin_root = 0;
    end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=begin_root; j<=end_root; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", j));
    }
  }
  return testSuccess;
}

struct testEngine scatterEngine = {
  ScatterGetBuffSize,
  ScatterRunTest
};

#pragma weak ncclTestEngine=scatterEngine

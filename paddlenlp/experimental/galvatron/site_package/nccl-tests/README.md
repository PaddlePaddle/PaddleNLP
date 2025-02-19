# NCCL Tests

These tests check both the performance and the correctness of [NCCL](http://github.com/nvidia/nccl) operations.

## Build

To build the tests, just type `make`.

If CUDA is not installed in /usr/local/cuda, you may specify CUDA\_HOME. Similarly, if NCCL is not installed in /usr, you may specify NCCL\_HOME.

```shell
$ make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

NCCL tests rely on MPI to work on multiple processes, hence multiple nodes. If you want to compile the tests with MPI support, you need to set MPI=1 and set MPI\_HOME to the path where MPI is installed.

```shell
$ make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

## Usage

NCCL tests can run on multiple processes, multiple threads, and multiple CUDA devices per thread. The number of process is managed by MPI and is therefore not passed to the tests as argument. The total number of ranks (=CUDA devices) will be equal to (number of processes)\*(number of threads)\*(number of GPUs per thread).

### Quick examples

Run on 8 GPUs (`-g 8`), scanning from 8 Bytes to 128MBytes :
```shell
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

Run with MPI on 10 processes (potentially on multiple nodes) with 4 GPUs each, for a total of 40 GPUs:
```shell
$ mpirun -np 10 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4
```

### Performance

See the [Performance](doc/PERFORMANCE.md) page for explanation about numbers, and in particular the "busbw" column.

### Arguments

All tests support the same set of arguments :

* Number of GPUs
  * `-t,--nthreads <num threads>` number of threads per process. Default : 1.
  * `-g,--ngpus <GPUs per thread>` number of gpus per thread. Default : 1.
* Sizes to scan
  * `-b,--minbytes <min size in bytes>` minimum size to start with. Default : 32M.
  * `-e,--maxbytes <max size in bytes>` maximum size to end at. Default : 32M.
  * Increments can be either fixed or a multiplication factor. Only one of those should be used
    * `-i,--stepbytes <increment size>` fixed increment between sizes. Default : 1M.
    * `-f,--stepfactor <increment factor>` multiplication factor between sizes. Default : disabled.
* NCCL operations arguments
  * `-o,--op <sum/prod/min/max/avg/all>` Specify which reduction operation to perform. Only relevant for reduction operations like Allreduce, Reduce or ReduceScatter. Default : Sum.
  * `-d,--datatype <nccltype/all>` Specify which datatype to use. Default : Float.
  * `-r,--root <root/all>` Specify which root to use. Only for operations with a root like broadcast or reduce. Default : 0.
* Performance
  * `-n,--iters <iteration count>` number of iterations. Default : 20.
  * `-w,--warmup_iters <warmup iteration count>` number of warmup iterations (not timed). Default : 5.
  * `-m,--agg_iters <aggregation count>` number of operations to aggregate together in each iteration. Default : 1.
  * `-a,--average <0/1/2/3>` Report performance as an average across all ranks (MPI=1 only). <0=Rank0,1=Avg,2=Min,3=Max>. Default : 1.
* Test operation
  * `-p,--parallel_init <0/1>` use threads to initialize NCCL in parallel. Default : 0.
  * `-c,--check <check iteration count>` perform count iterations, checking correctness of results on each iteration. This can be quite slow on large numbers of GPUs. Default : 1.
  * `-z,--blocking <0/1>` Make NCCL collective blocking, i.e. have CPUs wait and sync after each collective. Default : 0.
  * `-G,--cudagraph <num graph launches>` Capture iterations as a CUDA graph and then replay specified number of times. Default : 0.

## Copyright

NCCL tests are provided under the BSD license. All source code and accompanying documentation is copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.


// reference: https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.14/cpp/tensorrt_llm/kernels/customAllReduceKernels.h

#include <cassert>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <vector>
#include <string>
#include <mutex>
#include "cuda_ipc_utils.h"
#include "helper.h"
#include "trt_reduce_internal.cuh"

using namespace trt_llm;


#define CHECK_ERR(cmd) do { \
    if (cmd == -1) { \
        perror("Error: "); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

using fptr_t = int64_t;

class AllReduceMeta {
 public:
  AllReduceMeta(int64_t rank_id, int64_t world_size, const std::vector<fptr_t>& buffers,
                const std::vector<fptr_t>& barrier_in, const std::vector<fptr_t>& barrier_out) {
    this->rank_id = (int)rank_id;
    this->world_size = (int)world_size;
    this->buffers = buffers;
    this->barrier_in = barrier_in;
    this->barrier_out = barrier_out;
  }

 public:
  int world_size;
  int rank_id;
  std::vector<fptr_t> buffers;
  std::vector<fptr_t> barrier_in;
  std::vector<fptr_t> barrier_out;
  int barrier_flag = 1;

  static std::shared_ptr<AllReduceMeta> instance;
};

std::shared_ptr<AllReduceMeta> AllReduceMeta::instance = nullptr;



// Get the number of bits for a given data type.
inline int get_bits(paddle::DataType dtype) {
  switch (dtype) {
    case paddle::DataType::FLOAT32:
      return 32;
    case paddle::DataType::BFLOAT16:
    case paddle::DataType::FLOAT16:
      return 16;
    default:
      assert(false && "Unsupported data type");
  }
}

// Check if customized all-reduce kernels can be applied.
inline bool CanApplyCustomAllReduce(int64_t num_elements, paddle::DataType dtype) {
  // The customized all-reduce kernel has the following requirement(s).
  return num_elements % (16 / ((get_bits(dtype) + 7) / 8)) == 0;
}

void init_custom_ar(int rank_id, int world_size) {
  const int buffer_max_size = 8 * 1024 * 1024;
  const int barrier_max_size = 8 * (24 + 2) * 8;
  static std::mutex init_mutex;
  std::lock_guard<std::mutex> lock(init_mutex);
  if (!AllReduceMeta::instance) {
    
    CUDA_CHECK(cudaSetDevice(rank_id));

    SharedMemory* shm = (rank_id == 0) ? create_shared_memory() : open_shared_memory();
    
    void* buffers_ptr = nullptr;
    checkCudaError(cudaMalloc(&buffers_ptr, buffer_max_size), "cudaMalloc ptr error");
    void* barrier_in_ptr = nullptr; 
    checkCudaError(cudaMalloc(&barrier_in_ptr, barrier_max_size), "cudaMalloc ptr error");
    void* barrier_out_ptr = nullptr;
    checkCudaError(cudaMalloc(&barrier_out_ptr, barrier_max_size), "cudaMalloc ptr error");

    checkCudaError(cudaIpcGetMemHandle(&shm->buffers_handles[rank_id], buffers_ptr), "Failed to get buffers IPC memory handle");
    checkCudaError(cudaIpcGetMemHandle(&shm->barrier_in_handles[rank_id], barrier_in_ptr), "Failed to get barrier_in IPC memory handle");
    checkCudaError(cudaIpcGetMemHandle(&shm->barrier_out_handles[rank_id], barrier_out_ptr), "Failed to get barrier_out IPC memory handle");
    shm->ready_flags[rank_id] = true;
    
    wait_for_ready_flags(shm, world_size);

    std::vector<fptr_t> buffers, barrier_in, barrier_out;

    for(int i=0; i< world_size; ++i){
      if(i == rank_id){
        buffers.emplace_back(reinterpret_cast<fptr_t>(buffers_ptr));
        barrier_in.emplace_back(reinterpret_cast<fptr_t>(barrier_in_ptr));
        barrier_out.emplace_back(reinterpret_cast<fptr_t>(barrier_out_ptr));
      }
      else{
        void* other_buffer_handle_data;
        CUDA_CHECK(cudaIpcOpenMemHandle(&other_buffer_handle_data, shm->buffers_handles[i], cudaIpcMemLazyEnablePeerAccess));
        buffers.emplace_back(reinterpret_cast<fptr_t>(other_buffer_handle_data));

        void* other_barrier_in_handle_data;
        CUDA_CHECK(cudaIpcOpenMemHandle(&other_barrier_in_handle_data, shm->barrier_in_handles[i], cudaIpcMemLazyEnablePeerAccess));
        barrier_in.emplace_back(reinterpret_cast<fptr_t>(other_barrier_in_handle_data));

        void* other_barrier_out_handle_data;
        CUDA_CHECK(cudaIpcOpenMemHandle(&other_barrier_out_handle_data, shm->barrier_out_handles[i], cudaIpcMemLazyEnablePeerAccess));
        buffers.emplace_back(reinterpret_cast<fptr_t>(other_barrier_out_handle_data));
      }
    }

    AllReduceMeta::instance = std::make_shared<AllReduceMeta>(rank_id, world_size, buffers, barrier_in, barrier_out);

  }
}

void dispose(fptr_t _fa) {
  auto fa = reinterpret_cast<AllReduceMeta*>(_fa);
  delete fa;
}

std::vector<paddle::Tensor> all_reduce(const paddle::Tensor& inp, int rank_id, int world_size) {
  auto out = paddle::empty_like(inp);

  if (!AllReduceMeta::instance) {
    init_custom_ar(rank_id, world_size);
  }
  auto m = AllReduceMeta::instance;
  auto stream = inp.stream();
  auto num_elements = inp.numel();
  auto dtype = inp.type();
  AllReduceStrategyType strategy = SelectImplementation(num_elements * ((get_bits(dtype) + 7) / 8), m->world_size);

  assert(strategy == AllReduceStrategyType::ONESHOT || strategy == AllReduceStrategyType::TWOSHOT);
  assert(CanApplyCustomAllReduce(num_elements, dtype));

  // Initialize the all-reduce kernel arguments.

  AllReduceParams params;
  params.ranks_per_node = world_size;
  params.rank = rank_id;
  params.local_rank = rank_id;
  params.local_input_buffer_ptr = const_cast<void *>(inp.data());
  params.local_output_buffer_ptr = out.data();
  params.elts_total = inp.numel();
  params.elts_size = get_bits(dtype) / 8;
  params.barrier_flag = ++(m->barrier_flag);

  for (int i = 0; i < world_size; ++i) {
    params.peer_comm_buffer_ptrs[i] = reinterpret_cast<void*>(m->buffers[i]);
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(m->barrier_in[i]);
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(m->barrier_out[i]);
  }

  auto data_type = out.type();
  trtCustomAllReduce(params, data_type, strategy, stream);
  return {out};
}

std::vector<paddle::Tensor> all_reduce_fake(const paddle::Tensor& inp,  int rank_id, int world_size) {
  auto out = paddle::empty_like(inp);
  return {out};
}

std::vector<std::vector<int64_t>> TrtReduceInferShape(
    const std::vector<int64_t>& inp_shape) {
  return {inp_shape};
}

std::vector<paddle::DataType> TrtReduceInferDtype(
    const paddle::DataType& inp_dtype) {
  return {inp_dtype};
}

PD_BUILD_OP(trt_reduce)
    .Inputs({"inp"})
    .Outputs({"output"})
    .Attrs({"rank_id: int", "world_size: int"})
    .SetKernelFn(PD_KERNEL(all_reduce))
    .SetInferShapeFn(PD_INFER_SHAPE(TrtReduceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TrtReduceInferDtype));
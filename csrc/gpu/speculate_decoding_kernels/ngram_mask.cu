#include <curand_kernel.h>
#include <cuda_fp16.h>
#include "cub/cub.cuh"
#include "paddle/extension.h"

#define STOP_LIST_BS 385

template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
public:
  typedef half DataType;
  typedef paddle::float16 data_t;
};

class TreeNode{
public:
    int token_id_;
    TreeNode *children_;
    int children_node_len_;

    TreeNode(): token_id_(-1), children_node_len_(0) {
        children_ = nullptr;
    }

    __host__ __device__ bool is_in_tree(TreeNode *node, int node_num, int token, int *idx) {
        for (int i=0; i < node_num; i++) {
            if (node->children_) {
                TreeNode *tmp = &node->children_[i];
                if(tmp->token_id_ == token) {
                    *idx = i;
                    return true;
                } else if(tmp->token_id_ != -1) {
                    continue;
                } else {
                    *idx = i;
                    break;
                }
            } else {
                *idx = i;
                break;
            }
        }
        return false;
    }

    __host__ __device__ void insert(const int *stop_list, const int stop_list_len) {
        TreeNode *node = this;
        for (int i=0; i < stop_list_len; i++) {
            int token = stop_list[i];
            if (token == -1) break;
            int idx;
            bool in_tree = is_in_tree(node, STOP_LIST_BS, token, &idx);
            if (!in_tree) {
                node->children_node_len_++;
                if (!node->children_) {
                    node->children_ = new TreeNode[STOP_LIST_BS];
                }
                node->children_[idx].token_id_ = token;
            }
            node = &(node->children_[idx]);
        }
    }

    __host__ __device__ void search(const int *prefix_token, int token_len, int *res, int *res_len) {
        TreeNode *node = this;
        int idx;
        for (int i=0; i < token_len; i++) {
            if (node) {
                bool in_tree = is_in_tree(node, node->children_node_len_, prefix_token[i], &idx);
                if (in_tree) {
                    node = &(node->children_[idx]);
                } else {
                    *res_len = 0;
                    return;
                }
            } else {
                break;
            }
        }
        if (node) {
            int id = 0;
            for (int i=0; i < node->children_node_len_; i++) {
                if(node->children_[i].token_id_ != -1) {
                    res[id++] = node->children_[i].token_id_;
                } else {
                    break;
                }
            }
            *res_len = id;
        }
    }

    void destroy_node(TreeNode *node) {
        if (node) {
            delete[] node;
        }
    }

    void destroy(TreeNode *head) {
        if (head->children_node_len_ == 0) {
            // last layer
            return;
        }
        for (int i=0; i < head->children_node_len_; i++) {
            destroy(&head->children_[i]);
        }
        destroy_node(head->children_);
    }
};


class TreeNodeGPU{
public:
    int token_id_;
    int children_node_len_;
    int next;

    __device__ void is_in_tree(TreeNodeGPU *node, TreeNodeGPU *head, int token, int *idx) {
        for (int i=0; i < node->children_node_len_; i++) {
            TreeNodeGPU *children = head + node->next + i;
            if (children->token_id_ == token) {
                *idx = node->next + i;
                return;
            }
        }
        *idx = -1;
    }

    __device__ void search(const int *prefix_token, int token_len, int *res, int *res_len) {
        TreeNodeGPU *node = this;
        int idx;
        for (int i=0; i < token_len; i++) {
            is_in_tree(node, this, prefix_token[i], &idx);
            if (idx != -1) {
                node = this + idx;
            } else {
                *res_len = 0;
                return;
            }
        }
        for (int i=0; i < node->children_node_len_; i++) {
            res[i] = (this + node->next + i)->token_id_;
        }
        res_len[0] = node->children_node_len_;
    }

    template <typename T>
    __device__ void search(const int *prefix_token, int token_len, T *logits) {
        TreeNodeGPU *node = this;
        int idx;
        for (int i=0; i < token_len; i++) {
            is_in_tree(node, this, prefix_token[i], &idx);
            if (idx != -1) {
                node = this + idx;
            } else {
                return;
            }
        }
        for (int i=0; i < node->children_node_len_; i++) {
            // printf("child token: %d\n", (this + node->next + i)->token_id_);
            logits[(this + node->next + i)->token_id_] = -10000.;
        }
    }
};

template <typename T>
__global__ void search_on_gpu(TreeNodeGPU *head, 
                              const int *input_sequences, 
                              T *logits,
                              const int logits_len,
                              const int max_input_len) {
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    if (ti < max_input_len) {
        int seq_offset = bi * max_input_len;
        const int *seq_this_thread = input_sequences + seq_offset + max_input_len - ti - 1;
        T *logits_this_thread = logits + bi * logits_len;
        head->search(seq_this_thread, ti + 1, logits_this_thread);
    }
}

__global__ void search_on_gpu(TreeNodeGPU *head, 
                              const int *input_sequences, 
                              int *res, 
                              int *res_len,
                              const int max_input_len) {
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    if (ti < max_input_len) {
        int seq_offset = bi * max_input_len;
        int res_offset = bi * max_input_len * STOP_LIST_BS + ti * STOP_LIST_BS;
        int res_len_offset = bi * max_input_len + ti;
        const int *seq_this_thread = input_sequences + seq_offset + max_input_len - ti - 1;
        int *res_this_thread = res + res_offset;
        int *res_len_this_thread = res_len + res_len_offset;
        head->search(seq_this_thread, ti + 1, res_this_thread, res_len_this_thread);
    }
}

template <typename T>
__global__ void set_value_reverse(const int *res,
                                  const int *res_len,
                                  T *logits,
                                  const int max_input_len,
                                  const int logits_len) {
    int bi = blockIdx.x;
    T *logits_now = logits + bi * logits_len;
    for (int i = threadIdx.x; i < logits_len; i += blockDim.x) {
        bool set_flag = true;
        const int *res_len_now = res_len + bi * max_input_len;
        for (int j = 0; j < max_input_len; j++) {
            const int *res_now = res + bi * max_input_len * STOP_LIST_BS + j * STOP_LIST_BS;
            for (int k = 0; k < res_len_now[j]; k++) {
                if (i == res_now[k]) {
                    set_flag = false;
                    break;
                }
            }
            if (!set_flag) break;
        }
        if (set_flag) logits_now[i] = -10000.;
    }
}

__global__ void setup_gpu_node(TreeNodeGPU *d_head, 
                               const int token_id, 
                               const int childre_node_num, 
                               const int now_id, 
                               const int next_id) {
    d_head[now_id].token_id_ = token_id;
    d_head[now_id].children_node_len_ = childre_node_num;
    d_head[now_id].next = next_id;
}

__global__ void print_kernel(TreeNodeGPU *d_node, int now_id) {
    printf("NodeGPU token_id: %d, next_id: %d, children_num: %d\n", 
            d_node[now_id].token_id_, d_node[now_id].next, d_node[now_id].children_node_len_);
}

void get_nodes_num(TreeNode *node, int& res) {
    res += node->children_node_len_;
    for (int i=0; i < node->children_node_len_; i++) {
        get_nodes_num(&(node->children_[i]), res);
    }
}

void setup_tree_cpu(TreeNode *head, const int *stop_list, int stop_list_len) {
    for (int i=0; i < STOP_LIST_BS; i++) {
        int offset = i * stop_list_len;
        head->insert(stop_list+offset, stop_list_len);
    }
}

void copy_tree(TreeNode *head, 
               TreeNodeGPU *d_head, 
               int depth, 
               int now_id, 
               int &next_id, 
               cudaStream_t stream) {
    if (!head) {
        return;
    }
    int tmp_next_id = next_id;
    if (head->children_node_len_ == 0) {
        tmp_next_id = -1;
    }
    setup_gpu_node<<<1, 1, 0, stream>>>(d_head, head->token_id_, head->children_node_len_, now_id, tmp_next_id);
    cudaDeviceSynchronize();
    depth++;
    int next_id_this_arrays = next_id;
    next_id += head->children_node_len_;
    for (int i=0; i < head->children_node_len_; i++) {
        int tmp_now_id = next_id_this_arrays + i;
        copy_tree(&(head->children_[i]), d_head, depth, tmp_now_id, next_id, stream);
    }
}

void search_on_cpu(TreeNode *head, 
                   const int *input_sequences, 
                   int *res, 
                   int *res_len, 
                   const int bs, 
                   const int max_input_len) {
    for (int i=0; i < bs; i++) {
        for (int j=0; j < max_input_len; j++) {
            int seq_offset = i * max_input_len;
            int res_offset = i * max_input_len * STOP_LIST_BS + j * STOP_LIST_BS;
            int res_len_offset = i * max_input_len + j;
            const int *seq_this_time = input_sequences + seq_offset + max_input_len - j - 1;
            int *res_this_time = res + res_offset;
            int *res_len_this_time = res_len + res_len_offset;
            head->search(seq_this_time, j + 1, res_this_time, res_len_this_time);
        }
    }
}


template <paddle::DataType D>
std::vector<paddle::Tensor> NgramMaskKernel(const paddle::Tensor& stop_list_tensor, 
                                            const paddle::Tensor& input_sequences,
                                            const paddle::Tensor& logits,
                                            int reverse) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    std::vector<int64_t> stop_list_shape = stop_list_tensor.shape();
    std::vector<int64_t> input_shape = input_sequences.shape();
    std::vector<int64_t> logits_shape = logits.shape();
    int logits_len = logits_shape[1];
    auto logits_out = logits.copy_to(logits.place(), false);
    auto cu_stream = input_sequences.stream();
    int bs = input_shape[0];
    int stop_list_len = stop_list_shape[1];
    int max_input_len = input_shape[1];
    static int run_flag = 0;
    static TreeNode *head = new TreeNode();
    static TreeNodeGPU *d_head;
    if (!run_flag) {
        setup_tree_cpu(head, stop_list_tensor.data<int>(), stop_list_len);
        int node_num = 0;
        get_nodes_num(head, node_num);
        node_num++;
        printf("node_num: %d\n", node_num);

        run_flag++;

        cudaMalloc(&d_head, node_num * sizeof(TreeNodeGPU));
        cudaDeviceSynchronize();
        int now_id = 0;
        int next_id = 1;
        copy_tree(head, d_head, 0, now_id, next_id, cu_stream);
        cudaDeviceSynchronize();
        
        head->destroy(head);
    }

    int grid_size = bs;
    int block_size = max_input_len;
    if (reverse) {
        auto out_ids = paddle::full({bs, max_input_len, STOP_LIST_BS}, 0, paddle::DataType::INT32, input_sequences.place());
        auto out_lens = paddle::full({bs, max_input_len}, 0, paddle::DataType::INT32, input_sequences.place());
        int grid_size = bs;
        int block_size = max_input_len;
        search_on_gpu<<<grid_size, block_size, 0, cu_stream>>>(d_head, 
                                                               input_sequences.data<int>(), 
                                                               out_ids.data<int>(), 
                                                               out_lens.data<int>(),
                                                               max_input_len);


        set_value_reverse<DataType_><<<grid_size, 256, 0, cu_stream>>>(
            out_ids.data<int>(),
            out_lens.data<int>(),
            reinterpret_cast<DataType_*>(const_cast<data_t*>(logits_out.data<data_t>())),
            max_input_len,
            logits_len
        );
    } else {
        search_on_gpu<DataType_><<<grid_size, block_size, 0, cu_stream>>>(d_head, 
            input_sequences.data<int>(), 
            reinterpret_cast<DataType_*>(const_cast<data_t*>(logits_out.data<data_t>())),
            logits_len, 
            max_input_len);
    }
    
    return {logits_out};
}

std::vector<paddle::Tensor> NgramMask(const paddle::Tensor& stop_list_tensor, 
                                      const paddle::Tensor& input_sequences,
                                      const paddle::Tensor& logits,
                                      int reverse) {
    switch (logits.type()) {
        case paddle::DataType::FLOAT16: {
            return NgramMaskKernel<paddle::DataType::FLOAT16>(
                stop_list_tensor,
                input_sequences,
                logits,
                reverse
            );
        }
        case paddle::DataType::FLOAT32: {
            return NgramMaskKernel<paddle::DataType::FLOAT32>(
                stop_list_tensor,
                input_sequences,
                logits,
                reverse
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16 and float32 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> NgramMaskInferShape(const std::vector<int64_t>& stop_list_tensor_shape, 
                                                      const std::vector<int64_t>& input_sequences_shape,
                                                      const std::vector<int64_t>& logits_shape) {
    return {logits_shape};
}

std::vector<paddle::DataType> NgramMaskInferDtype(const paddle::DataType& stop_list_tensor_dtype, 
                                                  const paddle::DataType& input_sequences_dtype,
                                                  const paddle::DataType& logits_dtype) {
return {logits_dtype};
}

PD_BUILD_OP(ngram_mask)
.Inputs({"stop_list_tensor", "input_sequences", "logits"})
.Outputs({"logits_out"})
.Attrs({"reverse: int"})
.SetKernelFn(PD_KERNEL(NgramMask))
.SetInferShapeFn(PD_INFER_SHAPE(NgramMaskInferShape))
.SetInferDtypeFn(PD_INFER_DTYPE(NgramMaskInferDtype));

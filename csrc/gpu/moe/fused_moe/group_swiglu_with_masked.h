



template <typename T, int VecSize>
__global__ void group_swiglu_with_masked_kernel(T* act_out, 
                                 const T* input, 
                                 const int64_t *token_nums_per_expert,
                                 const int64_t group_num, 
                                 const int64_t group_size,
                                 const int64_t hidden_dim) {
    int64_t global_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t num = group_num * group_size * hidden_dim;
    using LoadT = AlignedVector<T, VecSize>;
    LoadT src_vec0, src_vec1;
    LoadT res_vec;

    for (int64_t i = global_idx * VecSize; i < num; i += blockDim.x * gridDim.x * VecSize) {
        
        const int64_t row_id = i / hidden_dim;
        const int64_t col_id = i % hidden_dim;

        const int64_t r_offset = row_id * hidden_dim * 2 + col_id;

        const int64_t group_id = row_id / group_size;
        const int64_t row_id_within_group = row_id % group_size;

        if (row_id_within_group >= token_nums_per_expert[group_id]) continue;

        Load<T, VecSize>(&input[r_offset], &src_vec0);
        Load<T, VecSize>(&input[r_offset + hidden_dim], &src_vec1);

        for (int j = 0; j < VecSize; ++j) {
            float a = static_cast<float>(src_vec0[j]);
            float b = static_cast<float>(src_vec1[j]);
            float res = b * a / (1.f + exp(-a));
            res_vec[j] = static_cast<T>(res);
        }

        Store<T, VecSize>(res_vec, &act_out[i]);       
    }
}

paddle::Tensor group_swiglu_with_masked(const paddle::Tensor& fc1_out_tensor, 
                                                  const paddle::Tensor& token_nums_per_expert
                                                  )
{
    const int64_t group_num = token_nums_per_expert.shape()[0];
    const int64_t group_size = fc1_out_tensor.shape()[0] / group_num;
    const int64_t hidden_dim = fc1_out_tensor.shape()[1] / 2;
    auto act_out_tensor = paddle::empty({group_num * group_size, hidden_dim}, fc1_out_tensor.dtype(), fc1_out_tensor.place());

    
    constexpr paddle::DataType D = paddle::DataType::BFLOAT16;
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    
    const int threads = 512;
    const int blocks = 256;
    group_swiglu_with_masked_kernel<DataType_, 8><<<blocks, threads, 0, fc1_out_tensor.stream()>>>(

        reinterpret_cast<DataType_*>(const_cast<data_t*>(act_out_tensor.data<data_t>())),
        reinterpret_cast<const DataType_*>(fc1_out_tensor.data<data_t>()),
        token_nums_per_expert.data<int64_t>(),

        group_num,
        group_size,
        hidden_dim
    );

    return act_out_tensor;
}
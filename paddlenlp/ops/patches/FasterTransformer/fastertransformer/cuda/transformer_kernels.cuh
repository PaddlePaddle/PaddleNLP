

namespace fastertransformer {

template <typename T>
void t5_layer_norm(const T *from_tensor, const T *gamma,
                const T *beta, T *norm_from_tensor_buf_, const int m, const int n, cudaStream_t stream);

template <typename T>
void add_bias_input_t5_layernorm_2_kernelLauncher(const T* input,
                                               const T* gamma,
                                               const T* beta,
                                               const T* bias,
                                               T* output,
                                               T* norm_output,
                                               int m,
                                               int n,
                                               cudaStream_t stream);

}


namespace fastertransformer
{

template <typename T>
void add_fusedQKV_bias_transpose_kernelLauncher(
  T* q_buf,
  T* k_buf,
  T* v_buf,
  T* QKV,
  const T* qkv_bias,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  const int rotary_embedding_dim,
  cudaStream_t stream);


} // namespace fastertransformer
#include <cstdlib>
#include <cstdio>
#include <string>
#include "helper.h"
#include <sys/mman.h>
#include <fstream>

template <int THREADBLOCK_SIZE>
__global__ void GetMaxLenKernel(const int *seq_lens,
                                int *max_len,
                                const int batch_size) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<int, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int max_len_this_thread = 0;
  for (int i = tid; i < batch_size; i += blockDim.x) {
    max_len_this_thread = max(seq_lens[i], max_len_this_thread);
  }
  int total =
      BlockReduce(temp_storage).Reduce(max_len_this_thread, MaxOp<int>());
  if (tid == 0) {
    *max_len = total;
  }
}

// void write_mmap(const std::string& name, const std::string& value) {
//     int len = value.length();
//     // 打开文件
//     int fd = open(name.data(), O_RDWR | O_CREAT, 00777);
//     // lseek将文件指针往后移动 len - 1 位
//     lseek(fd, len - 1, SEEK_END);
//     // 预先写入一个空字符；mmap不能扩展文件长度，这里相当于预先给文件长度，准备一个空架子
//     write(fd, " ", 1);
//     // 建立映射
//     char *buffer = (char *) mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
//     // 关闭文件
//     close(fd);
//     // 将 data 复制到 buffer 里
//     memcpy(buffer, data, len);
//     // 关闭映射
//     munmap(buffer, len)
// }


void GetMaxLen(const paddle::Tensor& seq_lens_encoder, const paddle::Tensor& seq_lens_decoder) {
    constexpr int blockSize = 128;
    int batch_size = seq_lens_encoder.shape()[0];
    auto cu_stream = seq_lens_encoder.stream();

    auto max_len_encoder = paddle::empty({1}, paddle::DataType::INT32, seq_lens_encoder.place());
    auto max_len_decoder = paddle::empty({1}, paddle::DataType::INT32, seq_lens_encoder.place());

    
    GetMaxLenKernel<blockSize><<<1, blockSize, 0, cu_stream>>>(
        seq_lens_encoder.data<int>(), max_len_encoder.data<int>(), batch_size);
    GetMaxLenKernel<blockSize><<<1, blockSize, 0, cu_stream>>>(
        seq_lens_decoder.data<int>(), max_len_decoder.data<int>(), batch_size);


    int max_len_encoder_data = max_len_encoder.copy_to(paddle::CPUPlace(), true).data<int>()[0];
    int max_len_decoder_data = max_len_decoder.copy_to(paddle::CPUPlace(), true).data<int>()[0];


    // char tmp_1[10];
    // itoa(max_len_encoder_data, tmp_1, 10);
    // char tmp_2[10];
    // itoa(max_len_decoder_data, tmp_2, 10);

    // std::string max_len_encoder_str = std::to_string(max_len_encoder_data);
    // std::string max_len_decoder_str = std::to_string(max_len_decoder_data);

    // int s = setenv("FLAGS_max_enc_len_this_time_data", max_len_encoder_str.data(),1);
    // printf("set env %d\n", s);
    // s = setenv("FLAGS_max_dec_len_this_time_data", max_len_decoder_str.data(),1);

    // auto env = getenv("FLAGS_max_enc_len_this_time_data");
    // printf("get env %s\n", env);


    std::ofstream outfile;
    outfile.open("max_len.txt", std::ios::out);

    outfile << max_len_encoder_data << "\n" << max_len_decoder_data;

    outfile.close();
}


PD_BUILD_OP(get_max_len)
    .Inputs({"seq_lens_encoder", "seq_lens_decoder"})
    .Outputs({"seq_lens_encoder_out", "seq_lens_decoder_out"})
    .SetInplaceMap({{"seq_lens_encoder", "seq_lens_encoder_out"}, {"seq_lens_decoder", "seq_lens_decoder_out"}})
    .SetKernelFn(PD_KERNEL(GetMaxLen));
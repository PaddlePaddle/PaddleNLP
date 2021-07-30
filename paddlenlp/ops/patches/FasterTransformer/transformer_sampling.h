/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Decoder transformer
 **/

#pragma once

#include <cuda_runtime.h>
#include "fastertransformer/allocator.h"
#include "fastertransformer/arguments.h"
#include "fastertransformer/common.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/open_decoder.h"

namespace fastertransformer {

template <OperationType OpType_>
class TransformerSampling {
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const IAllocator &allocator_;
  struct TransformerSamplingArguments args_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[1] = {20};

  OpenTransformerDecoder<OpType_> *decoder_;
  DataType_ **K_cache_;
  DataType_ **V_cache_;
  DataType_ *from_tensor_[2];
  DataType_ *decoder_buf_;
  DataType_ *trans_out_buf_;
  DataType_ *decoder_normed_result_buf_;
  DataType_ *embedding_buf_;
  DataType_ *lm_normed_result_buf_;
  DataType_ *logits_buf_;
  int *word_ids_buf_;
  bool *finished_buf_;

  void *buf_;
  int *finished_count_buf_;
  bool *h_finished_buf_;

  void *topk_workspace_ = nullptr;
  size_t topk_workspace_size_ = 0;
  void *topp_workspace_ = nullptr;
  size_t topp_workspace_size_ = 0;
  int *topp_id_vals_buf_;
  int *topp_offset_buf_;

public:
  TransformerSampling(const IAllocator &allocator,
                      const int batch_size,
                      const int seq_len,
                      const int head_num,
                      const int size_per_head,
                      const int vocab_size,
                      const int decoder_layers,
                      const int memory_hidden_units,
                      const int memory_max_seq_len,
                      const int start_id,
                      const int end_id,
                      const int type_id,
                      const int candidate_num = 0,
                      const float probability_threshold = 0.0,
                      const bool normalization_before = true,
                      const int unk_id = -1,
                      const int mask_id = -1,
                      const float temperature = 1.0,
                      const float len_penalty = 1.0,
                      const float repeat_penalty = 1.0)
      : allocator_(allocator) {
    args_.batch_size_ = batch_size;
    args_.seq_len_ = seq_len;
    args_.start_len_ = memory_max_seq_len;
    args_.head_num_ = head_num;
    args_.size_per_head_ = size_per_head;
    args_.hidden_units_ = head_num * size_per_head;
    args_.decoder_layers_ = decoder_layers;
    args_.vocab_size_ = vocab_size;
    args_.candidate_num_ = candidate_num;
    args_.probability_threshold_ = probability_threshold;
    args_.start_id_ = start_id;
    args_.end_id_ = end_id;
    args_.unk_id_ = unk_id;
    args_.mask_id_ = mask_id;
    args_.temperature_ = temperature;
    args_.normalization_before_ = normalization_before;
    args_.type_id_ = type_id;
    args_.len_penalty = len_penalty;
    args_.repeat_penalty = repeat_penalty;

    if (args_.candidate_num_ == 0 && args_.probability_threshold_ == 0.0) {
      printf(
          "[ERROR] Candidate_num for topk is 0 and probability threshold for "
          "top p is 0.0 \n");
      exit(-1);
    } else if (args_.candidate_num_ != 0 &&
               args_.probability_threshold_ != 0.0) {
      printf(
          "[ERROR] Candidate_num for topk is not 0 and probability threshold "
          "for top p is not 0.0 \n");
      exit(-1);
    }
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    K_cache_ = new DataType_ *[1];
    V_cache_ = new DataType_ *[1];

    decoder_ = new OpenTransformerDecoder<OpType_>(batch_size,
                                                   memory_max_seq_len,
                                                   head_num,
                                                   size_per_head,
                                                   memory_hidden_units,
                                                   normalization_before);

    int from_tensor_size = args_.batch_size_ * args_.hidden_units_;  // type T
    int decoder_workspace_size = decoder_->getWorkspaceSize();       // type T
    int decoder_normed_result_buffer_size =
        args_.batch_size_ * args_.hidden_units_;  // type T
    int cache_size = args_.batch_size_ * (args_.seq_len_ + args_.start_len_) *
                     args_.hidden_units_;                         // type T
    int logits_buf_size = args_.batch_size_ * args_.vocab_size_;  // type T

    int word_ids_buf_size = args_.batch_size_;            // type int
    int finished_buf_size = args_.batch_size_;            // type bool
    int finished_count_size = (int)(ceil(1 / 32.)) * 32;  // type int

    int topp_id_vals_buf_size =
        args_.batch_size_ * args_.vocab_size_;         // type int
    int topp_offset_buf_size = args_.batch_size_ + 1;  // type int

    // prevent memory misalinged address
    logits_buf_size = (int)(ceil(logits_buf_size / 4.)) * 4;
    word_ids_buf_size = (int)(ceil(word_ids_buf_size / 4.)) * 4;
    finished_buf_size = (int)(ceil(finished_buf_size / 32.)) * 32;

    topp_id_vals_buf_size = (int)(ceil(topp_id_vals_buf_size / 4.)) * 4;
    topp_offset_buf_size = (int)(ceil(topp_offset_buf_size / 4.)) * 4;

    topP_sampling_kernel_kernelLauncher(topp_workspace_,
                                        topp_workspace_size_,
                                        logits_buf_,
                                        topp_id_vals_buf_,
                                        topp_offset_buf_,
                                        finished_buf_,
                                        0,
                                        args_,
                                        nullptr,
                                        nullptr,
                                        args_.vocab_size_,
                                        0);
    topK_sampling_kernel_kernelLauncher(topk_workspace_,
                                        topk_workspace_size_,
                                        logits_buf_,
                                        nullptr,
                                        nullptr,
                                        finished_buf_,
                                        0,
                                        args_,
                                        0);

    int datatype_buf_size = from_tensor_size * 2 + decoder_workspace_size +
                            cache_size * 2 * args_.decoder_layers_ +
                            decoder_normed_result_buffer_size * 3;

    buf_ = reinterpret_cast<void *>(allocator_.malloc(
        sizeof(DataType_) * (datatype_buf_size + logits_buf_size) +
        sizeof(int) * word_ids_buf_size + sizeof(bool) * finished_buf_size +
        sizeof(int) * finished_count_size +
        sizeof(int) * (topp_id_vals_buf_size + topp_offset_buf_size) +
        topp_workspace_size_ + topk_workspace_size_));

    from_tensor_[0] = (DataType_ *)buf_;
    from_tensor_[1] = (DataType_ *)(from_tensor_[0] + from_tensor_size);

    /* We use two-way buffer since we have to update KV buf at the end of each
     * step. */
    K_cache_[0] = from_tensor_[1] + from_tensor_size +
                  0 * cache_size * args_.decoder_layers_;
    V_cache_[0] = from_tensor_[1] + from_tensor_size +
                  1 * cache_size * args_.decoder_layers_;

    decoder_buf_ = V_cache_[0] + cache_size * args_.decoder_layers_;
    trans_out_buf_ = (decoder_buf_ + decoder_workspace_size);
    // Used for pre-norm.
    decoder_normed_result_buf_ =
        (trans_out_buf_ + decoder_normed_result_buffer_size);
    // Same as decoder_normed_result_buf_. Used for post-norm.
    embedding_buf_ = (trans_out_buf_ + decoder_normed_result_buffer_size);
    lm_normed_result_buf_ =
        (decoder_normed_result_buf_ + decoder_normed_result_buffer_size);
    logits_buf_ = lm_normed_result_buf_ + decoder_normed_result_buffer_size;
    word_ids_buf_ = (int *)(logits_buf_ + logits_buf_size);
    finished_buf_ = (bool *)(word_ids_buf_ + word_ids_buf_size);
    finished_count_buf_ = (int *)(finished_buf_ + finished_buf_size);
    topp_id_vals_buf_ = (int *)(finished_count_buf_ + finished_count_size);
    topp_offset_buf_ = (int *)(topp_id_vals_buf_ + topp_id_vals_buf_size);
    topp_workspace_ = (void *)(topp_offset_buf_ + topp_offset_buf_size);
    topk_workspace_ = (void *)(topp_workspace_ + topp_workspace_size_);

    h_finished_buf_ = new bool[finished_buf_size];

    FILE *fd = fopen("decoding_gemm_config.in", "r");
    int err = 0;
    if (fd != NULL) {
      // The file found.
      err = fscanf(fd, "%d", &cublasAlgo_[0]);
      fclose(fd);
    }
    if (err != 1) {
      // The file not found. Default algorithm will be used.
      if (Traits_::OpType == OperationType::FP32) {
        cublasAlgo_[0] = CUBLAS_GEMM_DEFAULT;
      } else {
        cublasAlgo_[0] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      }
    } else {
      // check that the gemm_config setting is runnable
      if (Traits_::OpType == OperationType::FP32) {
        if (cublasAlgo_[0] > CUBLAS_GEMM_ALGO23 ||
            cublasAlgo_[0] < CUBLAS_GEMM_DEFAULT) {
          // the algorithm is not for FP32
          printf("[ERROR] cuBLAS Algorithm %d is not used in FP32. \n",
                 (int)cublasAlgo_[0]);
          exit(-1);
        }
      } else {
        if (cublasAlgo_[0] > CUBLAS_GEMM_ALGO15_TENSOR_OP ||
            cublasAlgo_[0] < CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
          // the algorithm is not for FP16
          printf("[ERROR] cuBLAS Algorithm %d is not used in FP16. \n",
                 (int)cublasAlgo_[0]);
          exit(-1);
        }
      }
    }
  }

  void forward(const TransformerDecoderInitParam<DataType_> *param,
               DecodingInitParam<DataType_> decoding_params) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    const int m = args_.batch_size_;
    const int k = args_.hidden_units_;
    const int n = args_.vocab_size_;

    /*
      sequence_length initialize to 0
      finished: false
      word_ids: start_id_
    */

    if (args_.candidate_num_ != 0) {
      sampling_init_kernelLauncher(finished_buf_,
                                   decoding_params.sequence_length,
                                   word_ids_buf_,
                                   args_.start_id_,
                                   args_.batch_size_,
                                   decoding_params.stream);
    } else if (args_.probability_threshold_ != 0.0) {
      topp_initialization_kernelLauncher(finished_buf_,
                                         decoding_params.sequence_length,
                                         word_ids_buf_,
                                         topp_id_vals_buf_,
                                         topp_offset_buf_,
                                         args_.vocab_size_,
                                         args_,
                                         decoding_params.stream);
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    int cache_size = args_.batch_size_ * (args_.seq_len_ + args_.start_len_) *
                     args_.hidden_units_;  // type T

    for (int layer = 0; layer < args_.decoder_layers_; ++layer) {
      init_cache_kernel_launcher(param[layer].k_cache,
                                 param[layer].v_cache,
                                 decoding_params.memory_sequence_length,
                                 K_cache_[0] + layer * cache_size,
                                 V_cache_[0] + layer * cache_size,
                                 args_.head_num_,
                                 args_.size_per_head_,
                                 args_.start_len_,
                                 args_.batch_size_,
                                 1,
                                 decoding_params.stream);
    }
#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    for (int step = 1; step <= args_.seq_len_; ++step) {
      if (args_.normalization_before_) {
        embeddings_kernel_launcher(from_tensor_[0],
                                   decoding_params.embedding_table,
                                   decoding_params.position_encoding_table,
                                   decoding_params.type_table,
                                   decoding_params.memory_sequence_length,
                                   word_ids_buf_,
                                   args_.type_id_,
                                   step,
                                   args_.batch_size_,
                                   args_.hidden_units_,
                                   decoding_params.stream);
      } else {
        embeddings_kernel_launcher(embedding_buf_,
                                   decoding_params.embedding_table,
                                   decoding_params.position_encoding_table,
                                   decoding_params.type_table,
                                   decoding_params.memory_sequence_length,
                                   word_ids_buf_,
                                   args_.type_id_,
                                   step,
                                   args_.batch_size_,
                                   args_.hidden_units_,
                                   decoding_params.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        decoder_->decoder_norm1(embedding_buf_,
                                decoding_params.layernorm.gamma,
                                decoding_params.layernorm.beta,
                                from_tensor_[0],
                                m,
                                k);
      }
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      int from_id, out_id;
      for (int layer = 0; layer < args_.decoder_layers_; ++layer) {
        /*
          For the first layer (layer-0), from_id is 0. We also stored the
          embedding lookup
          result in from_tensor_[0]
        */
        from_id = layer & 0x1;
        out_id = 1 - from_id;

        /*
          We use one decoder_ object to process multiple decoder layers.

          At the beginning of each decoder layer, we initialize the decoder
          object
          with corresponding weights and decoder_buf_.

          The decoder_buf_ is reused.
        */
        decoder_->initialize(param[layer], decoder_buf_);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        decoder_->forward(from_tensor_[from_id],
                          nullptr,
                          K_cache_[0] + layer * cache_size,
                          V_cache_[0] + layer * cache_size,
                          nullptr,
                          nullptr,
                          decoding_params.memory_sequence_length,
                          from_tensor_[out_id],
                          step,
                          args_.start_len_,
                          false);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      }

      DataType_ alpha = (DataType_)1.0f;
      DataType_ beta = (DataType_)0.0f;

      if (args_.normalization_before_) {
        decoder_->decoder_norm1(from_tensor_[out_id],
                                decoding_params.layernorm.gamma,
                                decoding_params.layernorm.beta,
                                decoder_normed_result_buf_,
                                m,
                                k);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        // trans here
        check_cuda_error(
            cublasGemmEx(decoding_params.cublas_handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         k,
                         m,
                         k,
                         &alpha,
                         decoding_params.trans_kernel,
                         AType_,
                         k,
                         decoder_normed_result_buf_,
                         BType_,
                         k,
                         &beta,
                         trans_out_buf_,
                         CType_,
                         k,
                         computeType_,
                         static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
      } else {
        // trans here
        check_cuda_error(
            cublasGemmEx(decoding_params.cublas_handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         k,
                         m,
                         k,
                         &alpha,
                         decoding_params.trans_kernel,
                         AType_,
                         k,
                         from_tensor_[out_id],
                         BType_,
                         k,
                         &beta,
                         trans_out_buf_,
                         CType_,
                         k,
                         computeType_,
                         static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
      }
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
      // add bias decoding_params.trans_bias
      decoder_->add_bias_act(trans_out_buf_,
                             decoding_params.trans_bias,
                             m,
                             k,
                             decoding_params.stream,
                             ActivationType::GELU);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      decoder_->decoder_norm1(trans_out_buf_,
                              decoding_params.lm_layernorm.gamma,
                              decoding_params.lm_layernorm.beta,
                              lm_normed_result_buf_,
                              m,
                              k);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
      check_cuda_error(
          cublasGemmEx(decoding_params.cublas_handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       n,
                       m,
                       k,
                       &alpha,
                       decoding_params.embedding_kernel,
                       AType_,
                       n,
                       lm_normed_result_buf_,
                       BType_,
                       k,
                       &beta,
                       logits_buf_,
                       CType_,
                       n,
                       computeType_,
                       static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      if (decoding_params.logits_mask_T || args_.temperature_ != 1.0 ||
          args_.len_penalty != 1.0 || args_.repeat_penalty != 1.0) {
        // TODO(): repeat penalty vertification.
        apply_penalties_Launcher(step,
                                 logits_buf_,
                                 nullptr, /*current_ids*/
                                 nullptr, /*previous_ids*/
                                 nullptr, /*parent_ids*/
                                 args_.batch_size_,
                                 1,
                                 args_.vocab_size_,
                                 args_.end_id_,
                                 args_.temperature_,
                                 args_.len_penalty,
                                 args_.repeat_penalty,
                                 decoding_params.stream,
                                 decoding_params.logits_mask_T);
      }

      if (args_.candidate_num_ != 0) {
        // top k sampling
        update_logits_without_softmax(logits_buf_,
                                      decoding_params.embedding_bias_T,
                                      args_.end_id_,
                                      finished_buf_,
                                      m,
                                      n,
                                      decoding_params.stream);

        topK_sampling_kernel_kernelLauncher(
            topk_workspace_,
            topk_workspace_size_,
            logits_buf_,
            decoding_params.output_ids + (step - 1) * args_.batch_size_,
            decoding_params.sequence_length,
            finished_buf_,
            step,  // used as random number
            args_,
            decoding_params.stream);
      } else if (args_.probability_threshold_ != 0.0) {
        // top p sampling
        softmax_kernelLauncher(logits_buf_,
                               decoding_params.embedding_bias_T,
                               args_.end_id_,
                               finished_buf_,
                               m,
                               n,
                               decoding_params.stream);

        topP_sampling_kernel_kernelLauncher(
            topp_workspace_,
            topp_workspace_size_,
            logits_buf_,
            topp_id_vals_buf_,
            topp_offset_buf_,
            finished_buf_,
            step,
            args_,
            decoding_params.output_ids + (step - 1) * args_.batch_size_,
            decoding_params.sequence_length,
            n,
            decoding_params.stream);
      }

      word_ids_buf_ =
          decoding_params.output_ids + (step - 1) * args_.batch_size_;

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      // TODO Find a better method to check the is_finished
      cudaMemcpy(h_finished_buf_,
                 finished_buf_,
                 sizeof(bool) * args_.batch_size_,
                 cudaMemcpyDeviceToHost);
      int sum = 0;
      for (int i = 0; i < args_.batch_size_; i++) {
        sum += (int)h_finished_buf_[i];
      }
      if (sum == args_.batch_size_) break;
    }
  }

  virtual ~TransformerSampling() {
    delete[] K_cache_;
    delete[] V_cache_;
    delete[] h_finished_buf_;
    delete decoder_;
    allocator_.free(buf_);
  }
};

}  // namespace fastertransformer

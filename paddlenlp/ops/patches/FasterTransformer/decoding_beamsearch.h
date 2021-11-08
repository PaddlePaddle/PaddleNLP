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
class DecodingBeamsearch {
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const IAllocator &allocator_;
  struct DecodingBeamsearchArguments args_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[1] = {20};

  OpenDecoder<OpType_> *decoder_;
  DataType_ **K_cache_;
  DataType_ **V_cache_;
  DataType_ **K_mem_cache_;
  DataType_ **V_mem_cache_;
  DataType_ *from_tensor_[2];
  DataType_ *decoder_buf_;
  DataType_ *decoder_normed_result_buf_;
  DataType_ *embedding_buf_;
  float *logits_buf_;
  float *cum_log_buf_;
  int *word_ids_buf_;
  int *parent_ids_buf_;
  bool *finished_buf_;
  void *buf_;
  int *finished_count_buf_;
  bool *h_finished_buf_;
  int *h_trg_length_;
  float *temp_storage_;

  bool is_fuse_topk_softMax_;
  bool keep_alive_beam_;

  void *topK_kernel_workspace = nullptr;
  size_t topk_workspace_size_ = 0;

public:
  DecodingBeamsearch(const IAllocator &allocator,
                     const int batch_size,
                     const int beam_width,
                     const int seq_len,
                     const int head_num,
                     const int size_per_head,
                     const int vocab_size,
                     const int decoder_layers,
                     const int memory_hidden_units,
                     const int memory_max_seq_len,
                     const int start_id,
                     const int end_id,
                     const float beam_search_diversity_rate = -0.0f,
                     const bool is_fuse_topk_softMax = false,
                     const bool keep_alive_beam = false,
                     const float alpha = 0.6,
                     const bool normalization_before = true,
                     const int pos_offset = 0,
                     const ActivationType act = ActivationType::RELU)
      : allocator_(allocator),
        is_fuse_topk_softMax_(is_fuse_topk_softMax),
        keep_alive_beam_(keep_alive_beam) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    args_.batch_size_ = batch_size;
    args_.beam_width_ = beam_width;
    args_.seq_len_ = seq_len;
    args_.head_num_ = head_num;
    args_.size_per_head_ = size_per_head;
    args_.hidden_units_ = head_num * size_per_head;
    args_.decoder_layers_ = decoder_layers;
    args_.vocab_size_ = vocab_size;
    args_.start_id_ = start_id;
    args_.end_id_ = end_id;
    args_.beam_search_diversity_rate_ = beam_search_diversity_rate;
    args_.alpha_ = alpha;
    args_.normalization_before_ = normalization_before;
    args_.pos_offset_ = pos_offset;
    args_.act_ = act;

    if (args_.beam_width_ > 16 || args_.beam_width_ > MAX_K)
      is_fuse_topk_softMax_ = false;

    K_cache_ = new DataType_ *[2];
    V_cache_ = new DataType_ *[2];

    K_mem_cache_ = new DataType_ *[args_.decoder_layers_];
    V_mem_cache_ = new DataType_ *[args_.decoder_layers_];
    decoder_ = new OpenDecoder<OpType_>(batch_size * beam_width,
                                        memory_max_seq_len,
                                        head_num,
                                        size_per_head,
                                        memory_hidden_units,
                                        normalization_before,
                                        args_.act_);

    int from_tensor_size =
        args_.batch_size_ * args_.beam_width_ * args_.hidden_units_;  // type T
    int decoder_workspace_size = decoder_->getWorkspaceSize();        // type T
    int decoder_normed_result_buffer_size =
        args_.batch_size_ * args_.beam_width_ * args_.hidden_units_;  // type T
    int cache_size = args_.batch_size_ * args_.beam_width_ * args_.seq_len_ *
                     args_.hidden_units_;  // type T
    int mem_cache_size = args_.batch_size_ * args_.beam_width_ *
                         memory_max_seq_len * args_.hidden_units_;  // type T

    int logits_buf_size = args_.batch_size_ * args_.beam_width_ *
                          args_.vocab_size_;                       // type float
    int cum_log_buf_size = args_.batch_size_ * args_.beam_width_;  // type float
    int word_ids_buf_size = args_.batch_size_ * args_.beam_width_;  // type int
    int parent_ids_buf_size =
        keep_alive_beam_ ? word_ids_buf_size : 0;                   // type int
    int finished_buf_size = args_.batch_size_ * args_.beam_width_;  // type bool
    int finished_count_size = (int)(ceil(1 / 32.)) * 32;            // type int

    int storage_size_per_beam =
        2 * args_.beam_width_ +
        SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2);
    args_.temp_storage_size_ = args_.batch_size_ * args_.beam_width_ *
                               storage_size_per_beam;  // type float

    // When using separated alive and finish beam queues, some buffers size need
    // to be doubled to restore beam search intermedia results of both alive and
    // finish beams.
    if (keep_alive_beam_ == true) {
      // cumulated log-probs of finish beams and alive beams
      cum_log_buf_size += cum_log_buf_size;
      finished_buf_size += finished_buf_size;
      // Double the size of topk_tmp_id_buf, topk_tmp_val_buf, since we need
      // select the top 2*beam_width.
      args_.temp_storage_size_ +=
          ceil(args_.batch_size_ * args_.beam_width_ * args_.beam_width_ / 4.) *
          4 * 2;
// Double tmp_buffer since we need select the top 2*beam_width.
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
      args_.temp_storage_size_ +=
          ceil(args_.batch_size_ * args_.beam_width_ *
               SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K) / 4.) *
          4;
#endif
    }

    // prevent memory misalinged address
    logits_buf_size = (int)(ceil(logits_buf_size / 4.)) * 4;
    cum_log_buf_size = (int)(ceil(cum_log_buf_size / 4.)) * 4;
    word_ids_buf_size = (int)(ceil(word_ids_buf_size / 4.)) * 4;
    parent_ids_buf_size = (int)(ceil(parent_ids_buf_size / 4.)) * 4;
    finished_buf_size = (int)(ceil(finished_buf_size / 32.)) * 32;
    args_.temp_storage_size_ = (int)(ceil(args_.temp_storage_size_ / 4.)) * 4;
    // get workspace size of topk kernel
    if (keep_alive_beam_ == true)
      topK_update_kernelLauncher(topK_kernel_workspace,
                                 topk_workspace_size_,
                                 logits_buf_,
                                 finished_buf_,
                                 nullptr,
                                 word_ids_buf_,
                                 parent_ids_buf_,
                                 nullptr,
                                 nullptr,
                                 cum_log_buf_,
                                 0,
                                 args_,
                                 0);
    else
      topK_kernelLauncher(topK_kernel_workspace,
                          topk_workspace_size_,
                          logits_buf_,
                          word_ids_buf_,
                          args_,
                          0);
    int datatype_buf_size =
        from_tensor_size * 2 + decoder_workspace_size +
        (cache_size * 4 + mem_cache_size * 2) * args_.decoder_layers_ +
        decoder_normed_result_buffer_size;

    buf_ = reinterpret_cast<void *>(allocator_.malloc(
        sizeof(DataType_) * datatype_buf_size +
        sizeof(float) * (logits_buf_size + cum_log_buf_size) +
        sizeof(int) * (word_ids_buf_size + parent_ids_buf_size) +
        sizeof(bool) * finished_buf_size + topk_workspace_size_ +
        sizeof(float) * args_.temp_storage_size_ +  // should be always float
        sizeof(int) * finished_count_size));

    from_tensor_[0] = (DataType_ *)buf_;
    from_tensor_[1] = (DataType_ *)(from_tensor_[0] + from_tensor_size);

    for (int i = 0; i < args_.decoder_layers_; ++i) {
      K_mem_cache_[i] =
          from_tensor_[1] + from_tensor_size + i * mem_cache_size * 2;
      V_mem_cache_[i] = from_tensor_[1] + from_tensor_size +
                        i * mem_cache_size * 2 + mem_cache_size;
    }
    /* We use two-way buffer since we have to update KV buf at the end of each
     * step. */
    K_cache_[0] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                  0 * cache_size * args_.decoder_layers_;
    K_cache_[1] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                  1 * cache_size * args_.decoder_layers_;
    V_cache_[0] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                  2 * cache_size * args_.decoder_layers_;
    V_cache_[1] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                  3 * cache_size * args_.decoder_layers_;

    decoder_buf_ = V_cache_[1] + cache_size * args_.decoder_layers_;
    decoder_normed_result_buf_ = (decoder_buf_ + decoder_workspace_size);
    // Used for post-norm.
    embedding_buf_ = (decoder_buf_ + decoder_workspace_size);
    logits_buf_ = (float *)(decoder_normed_result_buf_ +
                            decoder_normed_result_buffer_size);
    cum_log_buf_ = (float *)(logits_buf_ + logits_buf_size);
    word_ids_buf_ = (int *)(cum_log_buf_ + cum_log_buf_size);
    parent_ids_buf_ = (int *)(word_ids_buf_ + word_ids_buf_size);
    finished_buf_ = (bool *)(parent_ids_buf_ + parent_ids_buf_size);
    temp_storage_ = (float *)(finished_buf_ + finished_buf_size);
    finished_count_buf_ = (int *)(temp_storage_ + args_.temp_storage_size_);
    topK_kernel_workspace = (void *)(finished_count_buf_ + finished_count_size);

    h_finished_buf_ = new bool[finished_buf_size];
    h_trg_length_ = new int[args_.batch_size_];

    FILE *fd = fopen("decoding_gemm_config.in", "r");
    int err = 0;
    if (fd == NULL)
      printf("[WARNING] decoding_gemm_config.in is not found\n");
    else {
      err = fscanf(fd, "%d", &cublasAlgo_[0]);
      fclose(fd);
    }
    if (err != 1) {
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

  void forward(const DecoderInitParam<DataType_> *param,
               DecodingInitParam<DataType_> decoding_params) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    const int m = args_.batch_size_ * args_.beam_width_;
    const int k = args_.hidden_units_;
    const int n = args_.vocab_size_;

    int min_trg_len = 0;
    int max_trg_len = 0;

    if (decoding_params.trg_word) {
      cudaMemcpy(h_trg_length_,
                 decoding_params.trg_length,
                 sizeof(int) * args_.batch_size_,
                 cudaMemcpyDeviceToHost);
      min_trg_len = h_trg_length_[0];
      max_trg_len = h_trg_length_[0];

      for (int i = 1; i < args_.batch_size_; ++i) {
        min_trg_len = std::min(min_trg_len, h_trg_length_[i]);
        max_trg_len = std::max(max_trg_len, h_trg_length_[i]);
      }
    }

    /*
      sequence_length initialize to 0
      finished: false
      word_ids: start_id_
      cum_log_probs (for eacm beam, the first element is 0). e.g., [0 -inf -inf
      -inf][0 -inf -inf -inf]
      cum_log_probs: If keep_alive_beam_ is true, the first alive element is 0.
    */
    if (keep_alive_beam_ == true) {
      init_kernelLauncher_v2(finished_buf_,
                             decoding_params.sequence_length,
                             word_ids_buf_,
                             cum_log_buf_,
                             args_.start_id_,
                             args_.batch_size_,
                             args_.beam_width_ * 2,
                             decoding_params.stream);
    } else {
      init_kernelLauncher(finished_buf_,
                          decoding_params.sequence_length,
                          word_ids_buf_,
                          cum_log_buf_,
                          args_.start_id_,
                          args_.batch_size_,
                          args_.beam_width_,
                          decoding_params.stream);
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

/*
  User can check the init by init_kernel_check.
  init_kernel_check will compare the results of GPU and CPU.
  Note that init_kernel_check contains init and uses do not need to call it
  again.
*/
// init_kernel_check(finished_buf_, decoding_params.sequence_length,
// word_ids_buf_, cum_log_buf_,
//                   start_id_, batch_size_, beam_width_,
//                   decoding_params.stream);
#endif
    int cache_size = m * args_.seq_len_ * args_.hidden_units_;  // type T

    for (int step = 1; step <= args_.seq_len_; ++step) {
      // we use two-way buffer
      int kv_cache_id = step & 0x1;
      if (args_.normalization_before_) {
        embedding_lookup_sine_position_encoding_kernel_launcher(
            from_tensor_[0],
            decoding_params.embedding_table,
            decoding_params.position_encoding_table +
                (step - 1) * args_.hidden_units_,
            word_ids_buf_,
            m,
            args_.hidden_units_,
            decoding_params.stream);
      } else {
        // TODO(gongenlei): Only support Bart temporarily.
        embedding_position_lookups_bart_kernel_launcher(
            embedding_buf_,
            decoding_params.embedding_table,
            decoding_params.position_encoding_table +
                (step - 1 + args_.pos_offset_) * args_.hidden_units_,
            word_ids_buf_,
            m,
            args_.hidden_units_,
            decoding_params.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        decoder_->initialize_stream(decoding_params.stream);
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
                          decoding_params.memory_tensor,
                          K_cache_[kv_cache_id] + layer * cache_size,
                          V_cache_[kv_cache_id] + layer * cache_size,
                          K_mem_cache_[layer],
                          V_mem_cache_[layer],
                          decoding_params.memory_sequence_length,
                          from_tensor_[out_id],
                          step,
                          true);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      }
      if (step > min_trg_len) {
        float alpha = (float)1.0f;
        float beta = (float)0.0f;

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
                           decoder_normed_result_buf_,
                           BType_,
                           k,
                           &beta,
                           logits_buf_,
                           CUDA_R_32F,
                           n,
#ifdef CUDA11_MODE
                           CUBLAS_COMPUTE_32F_PEDANTIC,
#else
                           CUDA_R_32F,
#endif
                           static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
        } else {
          // Post-norm
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
                           from_tensor_[out_id],
                           BType_,
                           k,
                           &beta,
                           logits_buf_,
                           CUDA_R_32F,
                           n,
#ifdef CUDA11_MODE
                           CUBLAS_COMPUTE_32F_PEDANTIC,
#else
                           CUDA_R_32F,
#endif
                           static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        // Beamsearch
        if (is_fuse_topk_softMax_ == true) {
          if (keep_alive_beam_ == true) {
            // Use separated alive and finish beam queues to avoid the decrease
            // of alive beams.
            topK_softMax_update(logits_buf_,
                                decoding_params.embedding_bias,
                                finished_buf_,
                                decoding_params.sequence_length,
                                word_ids_buf_,
                                parent_ids_buf_,
                                decoding_params.output_ids + (step - 1) * m * 2,
                                decoding_params.parent_ids + (step - 1) * m * 2,
                                cum_log_buf_,
                                reinterpret_cast<void *>(temp_storage_),
                                step,
                                args_,
                                decoding_params.stream);
          } else {
            topK_softMax(logits_buf_,
                         decoding_params.embedding_bias,
                         finished_buf_,
                         cum_log_buf_,
                         word_ids_buf_,
                         reinterpret_cast<void *>(temp_storage_),
                         args_,
                         decoding_params.stream);
#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif

            update_kernelLauncher_v2(
                finished_buf_,
                decoding_params.parent_ids + (step - 1) * m,
                decoding_params.sequence_length,
                word_ids_buf_,
                decoding_params.output_ids + (step - 1) * m,
                finished_count_buf_,
                args_,
                decoding_params.stream);
#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
          }

        } else {
          if (keep_alive_beam_ == true) {
            update_logits_v2(logits_buf_,
                             decoding_params.embedding_bias,
                             args_.end_id_,
                             finished_buf_,
                             m,
                             n,
                             decoding_params.stream);

            // Use separated alive and finish beam queues to avoid the decrease
            // of alive beams.
            topK_update_kernelLauncher(
                topK_kernel_workspace,
                topk_workspace_size_,
                logits_buf_,
                finished_buf_,
                decoding_params.sequence_length,
                word_ids_buf_,
                parent_ids_buf_,
                decoding_params.output_ids + (step - 1) * m * 2,
                decoding_params.parent_ids + (step - 1) * m * 2,
                cum_log_buf_,
                step,
                args_,
                decoding_params.stream);
          } else {
            update_logits(logits_buf_,
                          decoding_params.embedding_bias,
                          args_.end_id_,
                          finished_buf_,
                          m,
                          n,
                          decoding_params.stream);

#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());

/*
  User can check the update_logits by update_logits_kernel_check.
  update_logits_kernel_check will compare the results of GPU and CPU.
  Note that update_logits_kernel_check contains update_logits and uses do not
  need to call it again.
*/
// update_logits_kernel_check(logits_buf_, decoding_params.embedding_bias,
// args_.end_id_, finished_buf_, m, n, decoding_params.stream);
#endif

            /* adding cum_log_buf_ to logits_buf_ */
            broadcast_kernelLauncher(logits_buf_,
                                     cum_log_buf_,
                                     args_.batch_size_,
                                     args_.beam_width_,
                                     args_.vocab_size_,
                                     decoding_params.stream);
#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());

/*
  User can check the broadcast_kernel by broadcast_kernel_check.
  broadcast_kernel_check will compare the results of GPU and CPU.
  Note that broadcast_kernel_check contains broadcast_kernelLauncher and uses do
  not need to call it again.
*/
// broadcast_kernel_check(logits_buf_, cum_log_buf_, batch_size_, beam_width_,
// vocab_size_, decoding_params.stream);
#endif

            topK_kernelLauncher(topK_kernel_workspace,
                                topk_workspace_size_,
                                logits_buf_,
                                word_ids_buf_,
                                args_,
                                decoding_params.stream);
#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
            update_kernelLauncher(logits_buf_,
                                  cum_log_buf_,
                                  finished_buf_,
                                  decoding_params.parent_ids + (step - 1) * m,
                                  decoding_params.sequence_length,
                                  word_ids_buf_,
                                  decoding_params.output_ids + (step - 1) * m,
                                  args_.batch_size_,
                                  args_.beam_width_,
                                  args_.vocab_size_,
                                  decoding_params.stream,
                                  args_.end_id_,
                                  finished_count_buf_);
          }
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      }

      if (step <= max_trg_len) {
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        update_with_force_deocdingLauncher<float>(
            decoding_params.trg_word,
            decoding_params.trg_length,
            finished_buf_,
            word_ids_buf_,
            (step > min_trg_len) ? nullptr : decoding_params.sequence_length,
            (keep_alive_beam_) ? parent_ids_buf_ : nullptr,
            (keep_alive_beam_) ? decoding_params.parent_ids + (step - 1) * m * 2
                               : decoding_params.parent_ids + (step - 1) * m,
            (keep_alive_beam_) ? decoding_params.output_ids + (step - 1) * m * 2
                               : decoding_params.output_ids + (step - 1) * m,
            cum_log_buf_,
            keep_alive_beam_,
            args_.batch_size_,
            (keep_alive_beam_) ? args_.beam_width_ * 2 : args_.beam_width_,
            max_trg_len,
            step,
            decoding_params.stream);
      }

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      update_KV_cache_kernelLauncher(
          K_cache_,
          V_cache_,
          keep_alive_beam_ ? parent_ids_buf_
                           : decoding_params.parent_ids + (step - 1) * m,
          args_.batch_size_,
          args_.beam_width_,
          args_.hidden_units_,
          step,
          cache_size,
          args_.decoder_layers_,
          decoding_params.stream);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());

/*
  User can check the update_KV_cache by update_KV_cache_kernel_check.
  update_KV_cache_kernel_check will compare the results of GPU and CPU.
  Note that update_KV_cache_kernel_check contains update_KV_cache and uses do
  not need to call it again.
*/
// update_KV_cache_kernel_check(K_cache_, V_cache_, decoding_params.parent_ids +
// (step - 1) * batch_size_ * beam_width_, batch_size_, beam_width_,
// hidden_units_, step, cache_size, decoder_layers_, decoding_params.stream);
#endif

      if (step > max_trg_len) {
        // TODO Find a better method to check the is_finished
        int finish_size = (keep_alive_beam_) ? m * 2 : m;
        cudaMemcpy(h_finished_buf_,
                   finished_buf_,
                   sizeof(bool) * finish_size,
                   cudaMemcpyDeviceToHost);
        int sum = 0;
        for (int i = 0; i < finish_size; i++) {
          sum += (int)h_finished_buf_[i];
        }
        if (sum == finish_size) break;
      }
    }  // end for decoding step for llop
  }    // end of forward

  virtual ~DecodingBeamsearch() {
    delete[] K_cache_;
    delete[] V_cache_;
    delete[] K_mem_cache_;
    delete[] V_mem_cache_;
    delete[] h_finished_buf_;
    delete[] h_trg_length_;
    delete decoder_;
    allocator_.free(buf_);
  }
};

}  // namespace fastertransformer

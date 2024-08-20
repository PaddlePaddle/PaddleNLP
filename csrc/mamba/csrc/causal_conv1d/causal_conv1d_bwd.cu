/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <paddle/phi/common/data_type.h>
#include <paddle/extension.h>


#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_reduce.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "causal_conv1d.h"
#include "causal_conv1d_common.h"
#include "static_switch.h"

template<int kNThreads_, int kWidth_, bool kSiluAct_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_bwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr bool kSiluAct = kSiluAct_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts);
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangeRounds = sizeof(float) / sizeof(input_t);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : custom_max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts * (!kSiluAct ? 1 : kNExchangeRounds + 1);
    static constexpr int kSmemSize = custom_max({kSmemExchangeSize,
            int(sizeof(typename BlockReduceFloatT::TempStorage))}) + (kIsVecLoad ? 0 : kSmemIOSize);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_bwd_kernel(ConvParamsBwd params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr bool kSiluAct = Ktraits::kSiluAct;
    static constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);
    vec_t *smem_exchange_x = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize) + kNThreads * kNExchangeRounds;
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + dim_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + dim_id * params.weight_c_stride;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + dim_id * params.dout_c_stride;
    input_t *dx = reinterpret_cast<input_t *>(params.dx_ptr) + batch_id * params.dx_batch_stride
        + dim_id * params.dx_c_stride;
    float *dweight = reinterpret_cast<float *>(params.dweight_ptr) + dim_id * params.dweight_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[dim_id]);

    // Thread kNThreads - 1 will load the first elements of the next chunk so we initialize those to 0.
    if (tidx == 0) {
        if constexpr (!kSiluAct) {
            input_t zeros[kNElts] = {input_t(0)};
            smem_exchange[0] = reinterpret_cast<vec_t *>(zeros)[0];
        } else {
            float zeros[kNElts] = {input_t(0)};
            #pragma unroll
            for (int r = 0; r < kNExchangeRounds; ++r) {
                smem_exchange[r * kNThreads] = reinterpret_cast<vec_t *>(zeros)[r];
            }
        }
    }

    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = weight[i * params.weight_width_stride]; }

    float dweight_vals[kWidth] = {input_t(0)};
    float dbias_val = 0;

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    x += (n_chunks - 1) * kChunkSize;
    dout += (n_chunks - 1) * kChunkSize;
    dx += (n_chunks - 1) * kChunkSize;
    for (int chunk = n_chunks - 1; chunk >= 0; --chunk) {
        input_t x_vals_load[2 * kNElts] = {input_t(0)};
        input_t dout_vals_load[2 * kNElts] = {input_t(0)};
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(x), *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), (params.seqlen - chunk * kChunkSize) / kNElts);
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(dout), *reinterpret_cast<vec_t (*)[1]>(&dout_vals_load[0]), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(x, *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), params.seqlen - chunk * kChunkSize);
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(dout, *reinterpret_cast<input_t (*)[kNElts]>(&dout_vals_load[0]), params.seqlen - chunk * kChunkSize);
        }
        float dout_vals[2 * kNElts], x_vals[2 * kNElts];
        if constexpr (!kSiluAct) {
            __syncthreads();
            // Thread 0 don't write yet, so that thread kNThreads - 1 can read
            // the first elements of the next chunk.
            if (tidx > 0) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(dout_vals_load)[0]; }
            __syncthreads();
            reinterpret_cast<vec_t *>(dout_vals_load)[1] = smem_exchange[tidx < kNThreads - 1 ? tidx + 1 : 0];
            __syncthreads();
            // Now thread 0 can write the first elements of the current chunk.
            if (tidx == 0) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(dout_vals_load)[0]; }
            #pragma unroll
            for (int i = 0; i < 2 * kNElts; ++i) {
                dout_vals[i] = float(dout_vals_load[i]);
                x_vals[i] = float(x_vals_load[i]);
            }
        } else {
            if (tidx == 0 && chunk > 0) {
                if constexpr(kIsVecLoad) {
                    reinterpret_cast<vec_t *>(x_vals_load)[0] = reinterpret_cast<vec_t *>(x)[-1];
                } else {
                    #pragma unroll
                    for (int i = 0; i < kNElts; ++i) {
                        if (chunk * kChunkSize + i < params.seqlen) { x_vals_load[i] = x[-kNElts + i]; }
                    }
                }
            }
            __syncthreads();
            smem_exchange_x[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1];
            __syncthreads();
            if (tidx > 0) { reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange_x[tidx - 1]; }
            #pragma unroll
            for (int i = 0; i < 2 * kNElts; ++i) { x_vals[i] = float(x_vals_load[i]); }
            // Recompute the output
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                float out_val = bias_val;
                #pragma unroll
                for (int w = 0; w < kWidth; ++w) {
                    out_val += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
                }
                float out_sigmoid_val = 1.0f / (1.0f + expf(-out_val));
                dout_vals[i] = float(dout_vals_load[i]) * out_sigmoid_val
                               * (1.0f + out_val * (1.0f - out_sigmoid_val));
            }
            // Exchange the dout_vals. It's possible that we need to do 2 rounds of exchange
            // if input_t is 16 bits (since then we'd have 8 values of float)
            __syncthreads();
            // Thread 0 don't write yet, so that thread kNThreads - 1 can read
            // the first elements of the next chunk.
            if (tidx > 0) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    smem_exchange[r * kNThreads + tidx] = reinterpret_cast<vec_t *>(dout_vals)[r];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < kNExchangeRounds; ++r) {
                reinterpret_cast<vec_t *>(dout_vals)[kNExchangeRounds + r]
                    = smem_exchange[r * kNThreads + (tidx < kNThreads - 1 ? tidx + 1 : 0)];
            }
            __syncthreads();
            // Now thread 0 can write the first elements of the current chunk.
            if (tidx == 0) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    smem_exchange[r * kNThreads + tidx] = reinterpret_cast<vec_t *>(dout_vals)[r];
                }
            }
        }
        dout -= kChunkSize;
        x -= kChunkSize;

        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { dbias_val += dout_vals[i]; }

        float dx_vals[kNElts] = {input_t(0)};
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                dx_vals[i] += weight_vals[w] * dout_vals[i + kWidth - w - 1];
            }
        }

        input_t dx_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { dx_vals_store[i] = dx_vals[i]; }
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockStoreVecT(smem_store_vec).Store(reinterpret_cast<vec_t*>(dx), reinterpret_cast<vec_t (&)[1]>(dx_vals_store), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            typename Ktraits::BlockStoreT(smem_store).Store(dx, dx_vals_store, params.seqlen - chunk * kChunkSize);
        }
        dx -= kChunkSize;

        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                dweight_vals[w] += x_vals[kNElts + i] * dout_vals[i + kWidth - w - 1];
            }
        }
    }

    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        __syncthreads();
        dweight_vals[w] = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dweight_vals[w]);
        if (tidx == 0) {
            atomicAdd(&reinterpret_cast<float *>(dweight)[w * params.dweight_width_stride], dweight_vals[w]);
        }
    }
    if (params.bias_ptr != nullptr) {
        __syncthreads();
        dbias_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dbias_val);
        if (tidx == 0) {
            atomicAdd(&reinterpret_cast<float *>(params.dbias_ptr)[dim_id], dbias_val);
        }
    }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_bwd_launch(ConvParamsBwd &params, cudaStream_t stream) {
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    BOOL_SWITCH(params.seqlen % kNElts == 0, kIsVecLoad, [&] {
        BOOL_SWITCH(params.silu_activation, kSiluAct, [&] {
            using Ktraits = Causal_conv1d_bwd_kernel_traits<kNThreads, kWidth, kSiluAct, kIsVecLoad, input_t, weight_t>;
            constexpr int kSmemSize = Ktraits::kSmemSize;
            dim3 grid(params.batch, params.dim);
            auto kernel = &causal_conv1d_bwd_kernel<Ktraits>;

            if (kSmemSize >= 48 * 1024) {
                #ifndef USE_ROCM
                cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
                #else
                // There is a slight signature discrepancy in HIP and CUDA "FuncSetAttribute" function.
                cudaFuncSetAttribute(
                    (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
                std::cerr << "Warning (causal_conv1d bwd launch): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
                #endif
            }


            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        });
    });
}

template<typename input_t, typename weight_t>
void causal_conv1d_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_bwd_launch<128, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_bwd_launch<128, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_bwd_launch<128, 4, input_t, weight_t>(params, stream);
    }
}

template<int kNThreads_, int kWidth_, int kChunkSizeL_, bool kSiluAct_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_channellast_bwd_kernel_traits {
    // The cache line is 128 bytes, and we try to read 16 bytes per thread.
    // So we have 8 threads per "row", so 32 or 64 elements in the channel dimension.
    // That leaves 4 columns per warp, and so 16 columns per block (assuming each block has 128
    // threads). Each each load is 16 x 32|64 elements in the L x C dimensions.
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr bool kSiluAct = kSiluAct_;
    static constexpr int kNThreads = kNThreads_;
    static_assert(kNThreads % 32 == 0);
    static constexpr int kNWarps = kNThreads / 32;
    static constexpr int kWidth = kWidth_;
    static constexpr int kChunkSizeL = kChunkSizeL_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static constexpr int kNEltsPerRow = 128 / kNBytes;
    static constexpr int kNThreadsPerRow = kNEltsPerRow / kNElts;  // Always 8 for now
    static_assert(kNThreadsPerRow * kNBytes * kNElts == 128);
    static constexpr int kNColsPerWarp = 32 / kNThreadsPerRow;  // Always 4 for now
    static_assert(kNColsPerWarp * kNThreadsPerRow == 32);
    static constexpr int kNColsPerLoad = kNColsPerWarp * kNWarps;
    static constexpr int kNLoads = kChunkSizeL / kNColsPerLoad;
    static_assert(kNLoads * kNColsPerLoad == kChunkSizeL);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    // using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    // using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // static constexpr int kSmemSize = std::max({sizeof(typename BlockLoadT::TempStorage),
    //                                            sizeof(typename BlockStoreT::TempStorage)});
    // static constexpr int kSmemSize = kChunkSizeL * kNEltsPerRow * kNBytes;
};

template<typename Ktraits, bool kHasSeqIdx, bool kHasDfinalStates>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_channellast_bwd_kernel(ConvParamsBwd params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr bool kSiluAct = Ktraits::kSiluAct;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNWarp = Ktraits::kNWarps;
    constexpr int kNThreadsPerC = Ktraits::kNThreadsPerRow;
    constexpr int kLPerLoad = Ktraits::kNColsPerLoad;
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    __shared__ input_t dout_smem[kChunkSizeL + kWidth - 1][kChunkSizeC + kNElts];
    __shared__ input_t x_smem[kWidth - 1 + kChunkSizeL + kWidth - 1][kChunkSizeC + kNElts];

    const int batch_id = blockIdx.x;
    const int chunk_l_id = blockIdx.y;
    const int chunk_c_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.x_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr)
        + chunk_c_id * kChunkSizeC * params.weight_c_stride;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.dout_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    input_t *dx = reinterpret_cast<input_t *>(params.dx_ptr) + batch_id * params.dx_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.dx_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    float *dweight = reinterpret_cast<float *>(params.dweight_ptr)
        + chunk_c_id * kChunkSizeC * params.dweight_c_stride;
    int *seq_idx = !kHasSeqIdx ? nullptr : reinterpret_cast<int *>(params.seq_idx_ptr)
        + batch_id * params.seqlen + chunk_l_id * kChunkSizeL;
    input_t *initial_states = params.initial_states_ptr == nullptr || chunk_l_id > 0 ? nullptr
        : reinterpret_cast<input_t *>(params.initial_states_ptr) + batch_id * params.initial_states_batch_stride + l_idx * params.initial_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    input_t *dinitial_states = params.dinitial_states_ptr == nullptr || chunk_l_id > 0 ? nullptr
        : reinterpret_cast<input_t *>(params.dinitial_states_ptr) + batch_id * params.dinitial_states_batch_stride + l_idx * params.dinitial_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    input_t *dfinal_states = params.dfinal_states_ptr == nullptr ? nullptr
        : reinterpret_cast<input_t *>(params.dfinal_states_ptr) + batch_id * params.dfinal_states_batch_stride + chunk_c_id * kChunkSizeC;

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t dout_vals_load[kNElts] = {input_t(0)};
        input_t x_vals_load[kNElts] = {input_t(0)};
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(dout_vals_load)[0] = *reinterpret_cast<vec_t *>(dout + l * kLPerLoad * params.dout_l_stride);
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + l * kLPerLoad * params.x_l_stride);
        }
        reinterpret_cast<vec_t *>(dout_smem[l * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(dout_vals_load)[0];
        reinterpret_cast<vec_t *>(x_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }
    // Load the elements from the previous chunk or next chunk that are needed for convolution.
    if (l_idx < kWidth - 1) {
        input_t dout_vals_load[kNElts] = {input_t(0)};
        input_t x_vals_load[kNElts] = {input_t(0)};
        if ((chunk_l_id + 1) * kChunkSizeL + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(dout_vals_load)[0] = *reinterpret_cast<vec_t *>(dout + kChunkSizeL * params.dout_l_stride);
        }
        if (chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) >= 0
            && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x - (kWidth - 1) * params.x_l_stride);
        } else if (initial_states != nullptr
                   && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < 0
                   && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(initial_states);
        }
        reinterpret_cast<vec_t *>(dout_smem[kChunkSizeL + l_idx])[c_idx] = reinterpret_cast<vec_t *>(dout_vals_load)[0];
        reinterpret_cast<vec_t *>(x_smem[l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }
    // Need to load (kWdith - 1) extra x's on the right to recompute the (kChunkSizeL + kWidth - 1) outputs
    if constexpr (kSiluAct) {
        if (l_idx < kWidth - 1) {
            input_t x_vals_load[kNElts] = {input_t(0)};
            if ((chunk_l_id + 1) * kChunkSizeL + l_idx < params.seqlen
                && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
                reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + kChunkSizeL * params.x_l_stride);
            }
            reinterpret_cast<vec_t *>(x_smem[kWidth - 1 + kChunkSizeL + l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
        }
    }

    __syncthreads();

    constexpr int kLPerThread = constexpr_min(kChunkSizeL * kChunkSizeC / kNThreads, kChunkSizeL);
    static_assert(kLPerThread * kNThreads == kChunkSizeL * kChunkSizeC);
    constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
    static_assert(kNThreadsPerRow * kLPerThread == kChunkSizeL);
    // kChunkSizeL, kLPerThread, kNThreadsPerRow should be powers of 2 for simplicity
    static_assert((kChunkSizeL & (kChunkSizeL - 1)) == 0);
    static_assert((kLPerThread & (kLPerThread - 1)) == 0);
    static_assert((kNThreadsPerRow & (kNThreadsPerRow - 1)) == 0);
    static_assert(kNThreadsPerRow <= 32);

    const int row_idx = tid / kNThreadsPerRow;
    const int col_idx = tid % kNThreadsPerRow;

    float bias_val = params.bias_ptr == nullptr || chunk_c_id * kChunkSizeC + row_idx >= params.dim ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[chunk_c_id * kChunkSizeC + row_idx]);
    float weight_vals[kWidth] = {input_t(0)};
    if (chunk_c_id * kChunkSizeC + row_idx < params.dim) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            weight_vals[w] = weight[row_idx * params.weight_c_stride + w * params.weight_width_stride];
        }
    }
    float dout_vals[kLPerThread + kWidth - 1];
    float x_vals[kWidth - 1 + kLPerThread + kWidth - 1];
    #pragma unroll
    for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
        dout_vals[i] = float(dout_smem[col_idx * kLPerThread + i][row_idx]);
        x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
    }

    int seq_idx_thread[kWidth - 1 + kLPerThread + kWidth - 1];
    if constexpr (kHasSeqIdx) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread + kWidth - 1; ++i) {
            const int l_idx = chunk_l_id * kChunkSizeL + col_idx * kLPerThread + i - (kWidth - 1);
            seq_idx_thread[i] = l_idx >= 0 && l_idx < params.seqlen ? seq_idx[col_idx * kLPerThread + i - (kWidth - 1)] : -1;
        }
    }

    if constexpr (kSiluAct) {  // Recompute the output
        #pragma unroll
        for (int i = kWidth - 1 + kLPerThread; i < kWidth - 1 + kLPerThread + kWidth - 1; ++i) {
            x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
        }
        #pragma unroll
        for (int i = 0; i < kLPerThread + kWidth - 1; ++i) {
            float out_val = bias_val;
            const int seq_idx_cur = !kHasSeqIdx ? 0 : seq_idx_thread[i + kWidth - 1];
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                if constexpr (!kHasSeqIdx) {
                    out_val += weight_vals[w] * x_vals[i + w];
                } else {
                    out_val += seq_idx_thread[i + w] == seq_idx_cur ? weight_vals[w] * x_vals[i + w] : 0.f;
                }
            }
            float out_val_sigmoid = 1.f / (1.f + expf(-out_val));
            dout_vals[i] *= out_val_sigmoid * (1 + out_val * (1 - out_val_sigmoid));
        }
    }

    float dweight_vals[kWidth] = {input_t(0)};
    SumOp<float> sum_op;
    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        #pragma unroll
        for (int i = 0; i < kLPerThread; ++i) {
            if constexpr (!kHasSeqIdx) {
                dweight_vals[w] += x_vals[i + w] * dout_vals[i];
            } else {
                dweight_vals[w] += seq_idx_thread[i + w] == seq_idx_thread[kWidth - 1 + i] ? x_vals[i + w] * dout_vals[i] : 0.f;
            }
        }
        dweight_vals[w] = Allreduce<kNThreadsPerRow>::run(dweight_vals[w], sum_op);
        if (col_idx == 0 && chunk_c_id * kChunkSizeC + row_idx < params.dim) {
            atomicAdd(&reinterpret_cast<float *>(dweight)[row_idx * params.dweight_c_stride + w * params.dweight_width_stride], dweight_vals[w]);
        }
    }

    if (params.bias_ptr != nullptr) {
        float dbias_val = 0.f;
        for (int i = 0; i < kLPerThread; ++i) { dbias_val += dout_vals[i]; }
        dbias_val = Allreduce<kNThreadsPerRow>::run(dbias_val, sum_op);
        if (col_idx == 0 && chunk_c_id * kChunkSizeC + row_idx < params.dim) {
            atomicAdd(&reinterpret_cast<float *>(params.dbias_ptr)[chunk_c_id * kChunkSizeC + row_idx], dbias_val);
        }
    }

    float dx_vals[kLPerThread] = {input_t(0)};
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        const int seq_idx_cur = !kHasSeqIdx ? 0 : seq_idx_thread[i + kWidth - 1];
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            if constexpr (!kHasSeqIdx) {
                dx_vals[i] += weight_vals[kWidth - 1 - w] * dout_vals[i + w];
            } else {
                dx_vals[i] += seq_idx_thread[kWidth - 1 + i + w] == seq_idx_cur ? weight_vals[kWidth - 1 - w] * dout_vals[i + w] : 0.f;
            }
        }
        // if (dfinal_states != nullptr) {
        if constexpr (kHasDfinalStates) {
            if (chunk_l_id * kChunkSizeL + col_idx * kLPerThread + i >= params.seqlen - kWidth + 1
                && chunk_l_id * kChunkSizeL + col_idx * kLPerThread + i < params.seqlen
                && chunk_c_id * kChunkSizeC + row_idx < params.dim) {
                dx_vals[i] += float(dfinal_states[((chunk_l_id * kChunkSizeL + col_idx * kLPerThread + i) - (params.seqlen - kWidth + 1)) * params.dfinal_states_l_stride + row_idx * params.dfinal_states_c_stride]);
            }
        }
    }

    float dxinit_vals[kWidth - 1] = {input_t(0)};
    static_assert(kLPerThread >= kWidth - 1);  // So only threads with col_idx == 0 need to handle dinitial_states
    if (dinitial_states != nullptr && col_idx == 0) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                dxinit_vals[i] += i + w - (kWidth - 1) >= 0 ? weight_vals[kWidth - 1 - w] * dout_vals[i + w - (kWidth - 1)] : 0.f;
            }
            // chunk_l_id must be 0 because dinitial_states != nullptr
            // if (dfinal_states != nullptr) {
            if constexpr (kHasDfinalStates) {
                if (i >= params.seqlen) {
                    dxinit_vals[i] += float(dfinal_states[(i - params.seqlen) * params.dfinal_states_l_stride + row_idx * params.dfinal_states_c_stride]);
                }
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) { x_smem[kWidth - 1 + col_idx * kLPerThread + i][row_idx] = dx_vals[i]; }
    if (dinitial_states != nullptr && col_idx == 0) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) { x_smem[i][row_idx] = dxinit_vals[i]; }
    }
    __syncthreads();

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t dx_vals_store[kNElts];
        reinterpret_cast<vec_t *>(dx_vals_store)[0] = reinterpret_cast<vec_t *>(x_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx];
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            *reinterpret_cast<vec_t *>(dx + l * kLPerLoad * params.dx_l_stride) = reinterpret_cast<vec_t *>(dx_vals_store)[0];
        }
    }
    if (dinitial_states != nullptr
        && l_idx < kWidth - 1
        && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
        input_t dxinit_vals_store[kNElts];
        reinterpret_cast<vec_t *>(dxinit_vals_store)[0] = reinterpret_cast<vec_t *>(x_smem[l_idx])[c_idx];
        *reinterpret_cast<vec_t *>(dinitial_states) = reinterpret_cast<vec_t *>(dxinit_vals_store)[0];
    }

}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_channellast_bwd_launch(ConvParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.silu_activation, kSiluAct, [&] {
        BOOL_SWITCH(params.seq_idx_ptr != nullptr, kHasSeqIdx, [&] {
            BOOL_SWITCH(params.dfinal_states_ptr != nullptr, kHasDfinalStates, [&] {
                BOOL_SWITCH(params.seqlen <= 128, kChunkSizeL64, [&] {
                    // kChunkSizeL = 128 is slightly faster than 64 when seqlen is larger
                    static constexpr int kChunk = kChunkSizeL64 ? 64 : 128;
                    using Ktraits = Causal_conv1d_channellast_bwd_kernel_traits<kNThreads, kWidth, kChunk, kSiluAct, true, input_t, weight_t>;
                    // constexpr int kSmemSize = Ktraits::kSmemSize;
                    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
                    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
                    const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
                    const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
                    dim3 grid(params.batch, n_chunks_L, n_chunks_C);
                    dim3 block(Ktraits::kNThreads);
                    auto kernel = &causal_conv1d_channellast_bwd_kernel<Ktraits, kHasSeqIdx, kHasDfinalStates>;
                    // if (kSmemSize >= 48 * 1024) {
                    //     C10_CUDA_CHECK(cudaFuncSetAttribute(
                    //         kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                    //     }
                    // kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                    kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
                });
            });
        });
    });
}

template<typename input_t, typename weight_t>
void causal_conv1d_channellast_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_channellast_bwd_launch<128, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_channellast_bwd_launch<128, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_channellast_bwd_launch<128, 4, input_t, weight_t>(params, stream);
    }
}

template void causal_conv1d_bwd_cuda<float, float>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<phi::dtype::float16, float>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<float, phi::dtype::float16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<phi::dtype::float16, phi::dtype::float16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_channellast_bwd_cuda<float, float>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_channellast_bwd_cuda<phi::dtype::float16, float>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_channellast_bwd_cuda<float, phi::dtype::float16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_channellast_bwd_cuda<phi::dtype::float16, phi::dtype::float16>(ConvParamsBwd &params, cudaStream_t stream);

#if defined(CUDA_BFLOAT16_AVAILABLE)
template void causal_conv1d_bwd_cuda<phi::dtype::bfloat16, float>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<phi::dtype::bfloat16, phi::dtype::float16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<float, phi::dtype::bfloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<phi::dtype::float16, phi::dtype::bfloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<phi::dtype::bfloat16, phi::dtype::bfloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_channellast_bwd_cuda<phi::dtype::bfloat16, float>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_channellast_bwd_cuda<phi::dtype::bfloat16, phi::dtype::float16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_channellast_bwd_cuda<float, phi::dtype::bfloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_channellast_bwd_cuda<phi::dtype::float16, phi::dtype::bfloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_channellast_bwd_cuda<phi::dtype::bfloat16, phi::dtype::bfloat16>(ConvParamsBwd &params, cudaStream_t stream);
#endif
/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "cutlass_unit_test.h"

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

using namespace cute;

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayout>> smem;
  cute::uint64_t tma_load_mbar[1];
};

// __grid_constant__ was introduced in CUDA 11.7.
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 7)))
#  define CUTE_GRID_CONSTANT_SUPPORTED
#endif

// __grid_constant__ can be enabled only on SM70+
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
#  define CUTE_GRID_CONSTANT_ENABLED
#endif

#if ! defined(CUTE_GRID_CONSTANT)
#  if defined(CUTE_GRID_CONSTANT_SUPPORTED) && defined(CUTE_GRID_CONSTANT_ENABLED)
#    define CUTE_GRID_CONSTANT __grid_constant__
#  else
#    define CUTE_GRID_CONSTANT
#  endif
#endif

#if CUDA_12_0_SM90_FEATURES_SUPPORTED
template <class T, class TiledCopy, class GmemLayout, class SmemLayout>
__global__ void
tma_test_device_cute(T const* g_in, T* g_out,
                     CUTE_GRID_CONSTANT TiledCopy const tma,
                     GmemLayout gmem_layout, SmemLayout smem_layout)
{
  assert(product_each(shape(gmem_layout)) == product_each(smem_layout.shape()));

  // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;
  // Construct SMEM tensor
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.data()), smem_layout);

#if 0

  //
  // Read in trivially
  //

  Tensor gA_in = make_tensor(make_gmem_ptr(g_in), gmem_layout);

  // Input gmem -> smem
  for (int i = threadIdx.x; i < size(sA); i += blockDim.x) {
    sA(i) = gA_in(i);
  }
  __syncthreads();

#else

  // TMA requires special handling of strides to deal with coord codomain mapping
  // Represent the full tensors -- get these from TMA
  Tensor gA = tma.get_tma_tensor(shape(gmem_layout));

  //
  // Prepare the TMA_LOAD
  //

  auto cta_tma = tma.get_slice(Int<0>{});        // CTA slice

  Tensor tAgA = cta_tma.partition_S(gA);         // (TMA,TMA_M,TMA_N)
  Tensor tAsA = cta_tma.partition_D(sA);         // (TMA,TMA_M,TMA_N)

#if 0
  if (thread0()) {
    print("  gA:  "); print(gA.data()); print(" o "); print(gA.layout()); print("\n");
    print("tAgA:  "); print(tAgA.data()); print(" o "); print(tAgA.layout()); print("\n");
    print("  sA:  "); print(sA.data()); print(" o "); print(sA.layout()); print("\n");
    print("tAsA:  "); print(tAsA.data()); print(" o "); print(tAsA.layout()); print("\n");
  }
#endif

  //
  // Perform the TMA_LOAD
  //

  // Group the TMA_M and TMA_N modes
  Tensor tAgA_2  = group_modes<1,rank(tAgA)>(tAgA);   // (TMA,Rest)
  Tensor tAsA_TR = group_modes<1,rank(tAsA)>(tAsA);   // (TMA,Rest)
  static_assert(size<1>(tAsA_TR) == 1);
  Tensor tAsA_2 = tAsA_TR(_,0);

  // Loop over the TMA stages, using smem as our buffer
  for (int stage = 0; stage < size<1>(tAgA_2); ++stage)
  {
    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    constexpr int kTmaTransactionBytes = size(sA) * sizeof(T);

    if (threadIdx.x == 0)
    {
      /// Initialize shared memory barrier
      tma_load_mbar[0] = 0;
      cute::initialize_barrier(tma_load_mbar[0], 1 /*numThreads*/);
      cute::set_barrier_transaction_bytes(tma_load_mbar[0], kTmaTransactionBytes);

      copy(tma.with(tma_load_mbar[0]), tAgA_2(_,stage), tAsA_2);
    }
    __syncthreads();

    /// Wait on the shared memory barrier until the phase bit flips from kPhaseBit value
    constexpr int kPhaseBit = 0;
    cute::wait_barrier(tma_load_mbar[0], kPhaseBit);

  #endif

    //
    // Write out trivially
    //

    Tensor gA_out = make_tensor(make_gmem_ptr(g_out), gmem_layout);
    // Do the same slicing and grouping as sA
    Tensor tAgA_out = cta_tma.partition_D(gA_out);         // (TMA,TMA_M,TMA_N)
    Tensor tAgA_2_out = group_modes<1,rank(tAgA_out)>(tAgA_out);   // (TMA,Rest)

    // Output smem -> gmem
    for (int i = threadIdx.x; i < size(tAsA_2); i += blockDim.x) {
      tAgA_2_out(i,stage) = tAsA_2(i);
    }
    __syncthreads();
  }
}

TEST(SM90_CuTe_Hopper, Tma_load_32x32_Col)
{
  using T = half_t;
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA  = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD 32x32 ColMajor SUCCESS\n");
}

TEST(SM90_CuTe_Hopper, Tma_load_32x32_Row)
{
  using T = half_t;
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};
  Layout gmem_layout = smem_layout;

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA  = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD 32x32 RowMajor SUCCESS\n");
}

TEST(SM90_CuTe_Hopper, Tma_load_GMMA_SW128_MN)
{
  using T = half_t;
  auto   smem_layout = GMMA::Layout_MN_SW128_Atom<T>{};
  Layout gmem_layout = make_layout(make_shape(size<0>(smem_layout), size<1>(smem_layout)), GenColMajor{});

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD GMMA::Layout_MN_SW128_Atom<T> SUCCESS\n");
}

TEST(SM90_CuTe_Hopper, Tma_load_GMMA_SW128_K)
{
  using T = half_t;
  auto   smem_layout = GMMA::Layout_K_SW128_Atom<T>{};
  Layout gmem_layout = make_layout(make_shape(size<0>(smem_layout), size<1>(smem_layout)), GenRowMajor{});

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD GMMA::Layout_K_SW128_Atom<T> SUCCESS\n");
}

TEST(SM90_CuTe_Hopper, Tma_load_GMMA_SW128_MN_Multi)
{
  using T = half_t;
  auto   smem_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, Shape<Int<128>,Int<128>>{});
  Layout gmem_layout = make_layout(make_shape(size<0>(smem_layout), size<1>(smem_layout)), GenColMajor{});

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD GMMA::Layout_MN_SW128_Atom<T> Multi SUCCESS\n");
}

TEST(SM90_CuTe_Hopper, Tma_load_GMMA_SW128_MN_Multi2)
{
  using T = half_t;
  // Tile the GMMA::Layout atom in the K-mode first, then the M-mode to get a bigger box size
  auto   smem_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, Shape<Int<128>,Int<128>>{}, Step<_2,_1>{});
  Layout gmem_layout = make_layout(make_shape(size<0>(smem_layout), size<1>(smem_layout)), GenColMajor{});

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD GMMA::Layout_MN_SW128_Atom<T> Multi SUCCESS\n");
}

TEST(SM90_CuTe_Hopper, Tma_load_GMMA_SW128_MN_Multi_Dyn)
{
  using T = half_t;
  auto   smem_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, Shape<Int<128>,Int<128>>{}, Step<_2,_1>{});
  Layout gmem_layout = make_layout(make_shape(128, 128), GenColMajor{});

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD GMMA::Layout_MN_SW128_Atom<T> Multi SUCCESS\n");
}

TEST(SM90_CuTe_Hopper, Tma_load_32x32_Multimode)
{
  using T = half_t;
  auto   smem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};
  Layout gmem_layout = make_layout(make_shape(make_shape(8,4), 32), GenRowMajor{});

  //auto   smem_layout = Layout<Shape<_32,_32>>{};
  //Layout gmem_layout = make_layout(make_shape(make_shape(8,4), 32), GenColMajor{});

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD GMMA::Layout_MN_SW128_Atom<T> Multi SUCCESS\n");
}

TEST(SM90_CuTe_Hopper, Tma_load_Tensor_blocking)
{
  using T = half_t;
  auto gmem_layout = make_shape(make_shape(336,40),make_shape(32,656));         // GMEM
  auto cta_tile    = make_shape(make_shape(_16{},_8{}),make_shape(_32{},_2{})); // GMEM Tiling:
                                                                                //   Take 16-elem from m0, 8-elem from m1,
                                                                                //   Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(cta_tile);                                     // Col-Major SMEM

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout, cta_tile, Int<1>{});
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD Tensor blocking SUCCESS\n");
}

TEST(SM90_CuTe_Hopper, Tma_load_Tensor_blocking_2)
{
  using T = half_t;
  auto gmem_layout = make_shape(make_shape(32,40),make_shape(make_shape(8,8),656)); // GMEM
  auto cta_tile    = make_shape(_128{},make_shape(_32{},_2{}));                // GMEM Tiling:
                                                                               //   Take 128-elem from m: m0 must divide 128,
                                                                               //                         m-last may be predicated
                                                                               //   Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(cta_tile);                                    // Col-Major SMEM

  thrust::host_vector<T> h_in(size(gmem_layout));
  for (int i = 0; i < h_in.size(); ++i) { h_in[i] = T(i); }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  Tensor gA = make_tensor(d_in.data().get(), gmem_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout, cta_tile, Int<1>{});
  //print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tma,
    gmem_layout,
    smem_layout);

  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe TMA_LOAD Tensor blocking 2 SUCCESS\n");
}
#endif

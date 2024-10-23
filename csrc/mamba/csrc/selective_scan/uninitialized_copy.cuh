/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#ifndef USE_ROCM
    #include <cub/config.cuh>

    #include <cuda/std/type_traits>
#else
    #include <hipcub/hipcub.hpp>
    // Map ::cuda::std to the standard std namespace
    namespace cuda {
        namespace std = ::std;
    }
#endif


namespace detail
{

#if defined(_NVHPC_CUDA)
template <typename T, typename U>
__host__ __device__ void uninitialized_copy(T *ptr, U &&val)
{
  // NVBug 3384810
  new (ptr) T(::cuda::std::forward<U>(val));
}
#else
template <typename T,
          typename U,
          typename ::cuda::std::enable_if<
            ::cuda::std::is_trivially_copyable<T>::value,
            int
          >::type = 0>
__host__ __device__ void uninitialized_copy(T *ptr, U &&val)
{
  *ptr = ::cuda::std::forward<U>(val);
}

template <typename T,
         typename U,
         typename ::cuda::std::enable_if<
           !::cuda::std::is_trivially_copyable<T>::value,
           int
         >::type = 0>
__host__ __device__ void uninitialized_copy(T *ptr, U &&val)
{
  new (ptr) T(::cuda::std::forward<U>(val));
}
#endif

} // namespace detail

// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#include <cute/tensor.hpp>
namespace cute {

////////////////////////////////////////////////////////////////////
// layout utils
////////////////////////////////////////////////////////////////////

// Permute layout based on indices, example:
//   permute_layout<1, 0>(layout) will swap the two dimensions
//   permute_layout<0, 2, 1>(layout) will swap the last two dimensions
template <size_t... I, typename Layout>
CUTE_HOST_DEVICE static constexpr auto permute_layout(Layout l) {
  static_assert(rank(l) == sizeof...(I), "Invalid permutation, rank mismatch");
  return cute::make_layout(cute::get<I>(l)...);
}

// is the layout f(x) = x
template <typename Layout>
CUTE_HOST_DEVICE static constexpr bool is_identity_layout() {
  if constexpr (std::is_same_v<Layout, void>) {
    return true;
  } else {
    constexpr auto coalesced_layout = coalesce(Layout{});
    if constexpr (rank(coalesced_layout) == 1 &&
                  stride<0>(coalesced_layout) == 1) {
      return true;
    }
    return false;
  }
}

////////////////////////////////////////////////////////////////////
// Pointer utils
////////////////////////////////////////////////////////////////////

template <class PointerType>
static constexpr auto get_logical_ptr(PointerType* ptr) {
  if constexpr (cute::sizeof_bits_v<PointerType> < 8) {
    return cute::subbyte_iterator<PointerType>(ptr);
  } else {
    return ptr;
  }
}

////////////////////////////////////////////////////////////////////
// Misc utils
////////////////////////////////////////////////////////////////////

template <typename T, typename Elements>
CUTE_HOST_DEVICE static constexpr auto create_auto_vectorizing_copy() {
  constexpr auto bits = sizeof_bits_v<T> * Elements{};
  if constexpr (bits % 128 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<128>{};
  } else if constexpr (bits % 64 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<64>{};
  } else if constexpr (bits % 32 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<32>{};
  } else if constexpr (bits % 16 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<16>{};
  } else {
    return AutoVectorizingCopyWithAssumedAlignment<8>{};
  }
}

};  // namespace cute

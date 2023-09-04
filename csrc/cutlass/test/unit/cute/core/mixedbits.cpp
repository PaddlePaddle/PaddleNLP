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

#include <cutlass/trace.h>
#include <cute/swizzle.hpp>

TEST(CuTe_core, MixedBits) {
  using namespace cute;

  auto uzero = cute::integral_constant<uint32_t, 0>{};

  for_each(make_integer_sequence<uint32_t, 8>{}, [&](auto S0) {
    for_each(make_integer_sequence<uint32_t, 8>{}, [&](auto F0) {
      for_each(make_integer_sequence<uint32_t, 8>{}, [&](auto S1) {
        for_each(make_integer_sequence<uint32_t, 8>{}, [&](auto F1) {
          if constexpr (decltype(S0 == uzero || S1 == uzero)::value) {
            return;
          } else if constexpr (decltype((S0 & F0) != uzero || (S1 & F1) != uzero)::value) {
            return;
          } else {
            for (uint32_t d0 = 0; d0 < 8; ++d0) {
              if ((d0 & F0) != d0) { continue; }    // Skip repeats
              for (uint32_t d1 = 0; d1 < 8; ++d1) {
                if ((d1 & F1) != d1) { continue; }  // Skip repeats
                auto m0 = make_mixed_bits(S0, d0, F0);
                auto m1 = make_mixed_bits(S1, d1, F1);
                //print(m0); print(" & "); print(m1); print(" = "); print(m0 & m1); print("\n");
                EXPECT_EQ(to_integral(m0 & m1), to_integral(m0) & to_integral(m1));
                //print(m0); print(" | "); print(m1); print(" = "); print(m0 | m1); print("\n");
                EXPECT_EQ(to_integral(m0 | m1), to_integral(m0) | to_integral(m1));
                //print(m0); print(" ^ "); print(m1); print(" = "); print(m0 ^ m1); print("\n");
                EXPECT_EQ(to_integral(m0 ^ m1), to_integral(m0) ^ to_integral(m1));
              }
            }
          }
        });
      });
    });
  });
}

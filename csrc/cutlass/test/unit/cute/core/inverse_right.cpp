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
#include <iostream>

#include <cute/tensor.hpp>

using namespace cute;

template <class Layout>
void
test_right_inverse(Layout const& layout)
{
  auto inv_layout = right_inverse(layout);

  CUTLASS_TRACE_HOST(layout << " ^ -1\n" << "  =>  \n" << inv_layout);
  CUTLASS_TRACE_HOST("Composition: " << coalesce(composition(layout, inv_layout)) << std::endl);

  for (int i = 0; i < size(inv_layout); ++i) {
    //printf("%3d: %3d  %3d\n", i, int(inv_layout(i)), int(layout(inv_layout(i))));
    EXPECT_EQ(layout(inv_layout(i)),  i);
  }
}

TEST(CuTe_core, Inverse_right)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("RIGHT INVERSE"                  );
  CUTLASS_TRACE_HOST("-------------------------------");

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("Simple tests"                   );
  CUTLASS_TRACE_HOST("-------------------------------");

  {
  auto layout = Layout<_1, _0>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<_1, _1>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_4>,
                       Stride<_0>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <Shape <_1,_1>>,
                       Stride<Stride<_0,_0>>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <Shape <_3,_7>>,
                       Stride<Stride<_0,_0>>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_1>,
                       Stride<_1>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_4>,
                       Stride<_1>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_4>,
                       Stride<_2>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_2,_4>,
                       Stride<_0,_2>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_8, _4>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_8, _4>,
                       Stride<_4, _1>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape< _2,_4,_6>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_2,_4,_6>,
                       Stride<_4,_1,_8>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_2,_4,_4,_6>,
                       Stride<_4,_1,_0,_8>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_4, _2>,
                       Stride<_1,_16>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = Layout<Shape <_4, _2>,
                       Stride<_1, _5>>{};

  test_right_inverse(layout);
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("Dynamic shapes/strides"         );
  CUTLASS_TRACE_HOST("-------------------------------");

  {
  auto layout = make_layout(Shape<_4, _2>{}, make_stride(Int<1>{}, 4));

  test_right_inverse(layout);
  }

  {
  auto layout = make_layout(make_shape(_4{}, 2), make_stride(Int<1>{}, 4));

  test_right_inverse(layout);
  }

  {
  auto layout = make_layout(make_shape(4, 2), make_stride(Int<1>{}, 4));

  test_right_inverse(layout);
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("Swizzle layouts"                );
  CUTLASS_TRACE_HOST("-------------------------------");

  {
  auto layout = ComposedLayout<Swizzle<1,0,2>, _0, Layout<Shape <_4, _4>,
                                                          Stride<_1, _4>>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = ComposedLayout<Swizzle<1,0,2>, _0, Layout<Shape <_4, _4>,
                                                          Stride<_4, _1>>>{};

  test_right_inverse(layout);
  }

  {
  auto layout = ComposedLayout<Swizzle<1,0,1>, _0, Layout<Shape <_4, _4>,
                                                          Stride<_8, _1>>>{};

  test_right_inverse(layout);
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("BETA: Negative strides"         );
  CUTLASS_TRACE_HOST("-------------------------------");

  // Negative strides (beta support)
  // Post-conditions/layout indexing aren't generalized enough to support these yet
  // However, the composition post-condition is general enough.
  {
  auto layout = make_layout(Shape<_4>{}, Stride<Int<-1>>{});

  test_right_inverse(layout);
  }

  //{
  //auto layout = Layout<Shape < _2,_4>,
  //                     Stride<_m1,_2>>{};

  //test_right_inverse(layout);
  //}

  //{
  //auto layout = Layout<Shape < _2, _4>,
  //                     Stride< _4,_m1>>{};

  //test_right_inverse(layout);
  //}

  //{
  //auto layout = Layout<Shape < _2, _4, _6>,
  //                     Stride<_m1,_12,_m2>>{};

  //test_right_inverse(layout);
  //}

}

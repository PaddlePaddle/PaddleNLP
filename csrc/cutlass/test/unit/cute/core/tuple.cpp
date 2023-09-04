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
#include <cute/tensor.hpp>

TEST(CuTe_core, Tuple)
{
  using namespace cute;

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("SIMPLE STATIC AND DYNAMIC TUPLES");
  CUTLASS_TRACE_HOST("-------------------------------");

  using tuple_2d_s_type = tuple<_8, _4>;                            // (8,4)
  using tuple_3d_s_type = tuple<_8, _4, _2>;                        // (8,4,2)
  using tuple_3h_s_type = tuple<tuple<_1, _2>, _8, _2>;             // ((1,2),8,2)

  using tuple_2d_d_type = tuple<int, int>;                          // (8,4)
  using tuple_3d_d_type = tuple<int, int, int>;                     // (8,4,2)
  using tuple_3h_d_type = tuple<tuple<int, int>, int, int>;         // ((1,2),8,2)

  using tuple_2d_m_type = tuple<_8, int>;                           // (8,4)
  using tuple_3d_m_type = tuple<int, int, _2>;                      // (8,4,2)
  using tuple_3h_m_type = tuple<tuple<int, _2>, int, int>;          // ((1,2),8,2)

  tuple_2d_s_type tuple_2d_s;
  tuple_3d_s_type tuple_3d_s;
  tuple_3h_s_type tuple_3h_s;

  tuple_2d_d_type tuple_2d_d(8,4);
  tuple_3d_d_type tuple_3d_d(8,4,2);
  tuple_3h_d_type tuple_3h_d(tuple<int,int>(1,2),8,2);

  tuple_2d_m_type tuple_2d_m(_8{}, 4);
  tuple_3d_m_type tuple_3d_m(8,4,_2{});
  tuple_3h_m_type tuple_3h_m(tuple<int,_2>(1,_2{}),8,2);

  CUTLASS_TRACE_HOST(tuple_2d_s << (is_static<tuple_2d_s_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_2d_s_type));
  ASSERT_TRUE(is_static<tuple_2d_s_type>::value == true);
  ASSERT_TRUE(sizeof(tuple_2d_s_type) == 1);
  ASSERT_TRUE(std::is_empty<tuple_2d_s_type>::value);

  CUTLASS_TRACE_HOST(tuple_3d_s << (is_static<tuple_3d_s_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3d_s_type));
  ASSERT_TRUE(is_static<tuple_3d_s_type>::value == true);
  ASSERT_TRUE(sizeof(tuple_3d_s_type) == 1);
  ASSERT_TRUE(std::is_empty<tuple_3d_s_type>::value);

  CUTLASS_TRACE_HOST(tuple_3h_s << (is_static<tuple_3h_s_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3h_s_type));
  ASSERT_TRUE(is_static<tuple_3h_s_type>::value == true);
  ASSERT_TRUE(sizeof(tuple_3h_s_type) == 1);
  ASSERT_TRUE(std::is_empty<tuple_3h_s_type>::value);

  CUTLASS_TRACE_HOST(tuple_2d_d << (is_static<tuple_2d_d_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_2d_d_type));
  ASSERT_TRUE(is_static<tuple_2d_d_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_2d_d_type) == 8);
  ASSERT_TRUE(!std::is_empty<tuple_2d_d_type>::value);

  CUTLASS_TRACE_HOST(tuple_3d_d << (is_static<tuple_3d_d_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3d_d_type));
  ASSERT_TRUE(is_static<tuple_3d_d_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_3d_d_type) == 12);
  ASSERT_TRUE(!std::is_empty<tuple_3d_d_type>::value);

  CUTLASS_TRACE_HOST(tuple_3h_d << (is_static<tuple_3h_d_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3h_d_type));
  ASSERT_TRUE(is_static<tuple_3h_d_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_3h_d_type) == 16);
  ASSERT_TRUE(!std::is_empty<tuple_3h_d_type>::value);

  CUTLASS_TRACE_HOST(tuple_2d_m << (is_static<tuple_2d_m_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_2d_m_type));
  ASSERT_TRUE(is_static<tuple_2d_m_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_2d_m_type) == 4);
  ASSERT_TRUE(!std::is_empty<tuple_2d_m_type>::value);

  CUTLASS_TRACE_HOST(tuple_3d_m << (is_static<tuple_3d_m_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3d_m_type));
  ASSERT_TRUE(is_static<tuple_3d_m_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_3d_m_type) == 8);
  ASSERT_TRUE(!std::is_empty<tuple_3d_m_type>::value);

  CUTLASS_TRACE_HOST(tuple_3h_m << (is_static<tuple_3h_m_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3h_m_type));
  ASSERT_TRUE(is_static<tuple_3h_m_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_3h_m_type) == 12);
  ASSERT_TRUE(!std::is_empty<tuple_3h_m_type>::value);

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("SIMPLE TUPLE OPS");
  CUTLASS_TRACE_HOST("-------------------------------");

  CUTLASS_TRACE_HOST("product(" << tuple_2d_s << ") => " << product(tuple_2d_s));
  CUTE_STATIC_ASSERT_V(product(tuple_2d_s) == _32{});
  CUTLASS_TRACE_HOST("product(" << tuple_3d_s << ") => " << product(tuple_3d_s));
  CUTE_STATIC_ASSERT_V(product(tuple_3d_s) == _64{});
  CUTLASS_TRACE_HOST("product(" << tuple_3h_s << ") => " << product(tuple_3h_s));
  CUTE_STATIC_ASSERT_V(product(tuple_3h_s) == _32{});

  CUTLASS_TRACE_HOST("product(" << tuple_2d_d << ") => " << product(tuple_2d_d));
  ASSERT_TRUE(product(tuple_2d_d) == 32);
  CUTLASS_TRACE_HOST("product(" << tuple_3d_d << ") => " << product(tuple_3d_d));
  ASSERT_TRUE(product(tuple_3d_d) == 64);
  CUTLASS_TRACE_HOST("product(" << tuple_3h_d << ") => " << product(tuple_3h_d));
  ASSERT_TRUE(product(tuple_3h_d) == 32);

  CUTLASS_TRACE_HOST("product(" << tuple_2d_m << ") => " << product(tuple_2d_m));
  ASSERT_TRUE(product(tuple_2d_m) == 32);
  CUTLASS_TRACE_HOST("product(" << tuple_3d_m << ") => " << product(tuple_3d_m));
  ASSERT_TRUE(product(tuple_3d_m) == 64);
  CUTLASS_TRACE_HOST("product(" << tuple_3h_m << ") => " << product(tuple_3h_m));
  ASSERT_TRUE(product(tuple_3h_m) == 32);

  CUTLASS_TRACE_HOST("max(" << tuple_2d_s << ") => " << max(tuple_2d_s));
  CUTE_STATIC_ASSERT_V(max(tuple_2d_s) == _8{});
  CUTLASS_TRACE_HOST("max(" << tuple_3d_s << ") => " << max(tuple_3d_s));
  CUTE_STATIC_ASSERT_V(max(tuple_3d_s) == _8{});
  CUTLASS_TRACE_HOST("max(" << tuple_3h_s << ") => " << max(tuple_3h_s));
  CUTE_STATIC_ASSERT_V(max(tuple_3h_s) == _8{});

  CUTLASS_TRACE_HOST("max(" << tuple_2d_d << ") => " << max(tuple_2d_d));
  ASSERT_TRUE(max(tuple_2d_d) == 8);
  CUTLASS_TRACE_HOST("max(" << tuple_3d_d << ") => " << max(tuple_3d_d));
  ASSERT_TRUE(max(tuple_3d_d) == 8);
  CUTLASS_TRACE_HOST("max(" << tuple_3h_d << ") => " << max(tuple_3h_d));
  ASSERT_TRUE(max(tuple_3h_d) == 8);

  CUTLASS_TRACE_HOST("max(" << tuple_2d_m << ") => " << max(tuple_2d_m));
  ASSERT_TRUE(max(tuple_2d_m) == 8);
  CUTLASS_TRACE_HOST("max(" << tuple_3d_m << ") => " << max(tuple_3d_m));
  ASSERT_TRUE(max(tuple_3d_m) == 8);
  CUTLASS_TRACE_HOST("max(" << tuple_3h_m << ") => " << max(tuple_3h_m));
  ASSERT_TRUE(max(tuple_3h_m) == 8);

  // 2d s|d|m
  CUTLASS_TRACE_HOST("inner_product(" << tuple_2d_s << ", " << tuple_2d_s << ") => "
            << inner_product(tuple_2d_s, tuple_2d_s));
  CUTE_STATIC_ASSERT_V(inner_product(tuple_2d_s, tuple_2d_s) == Int<80>{});
  CUTLASS_TRACE_HOST("inner_product(" << tuple_2d_d << ", " << tuple_2d_d << ") => "
            << inner_product(tuple_2d_d, tuple_2d_d));
  ASSERT_TRUE(inner_product(tuple_2d_d, tuple_2d_d) == 80);
  CUTLASS_TRACE_HOST("inner_product(" << tuple_2d_m << ", " << tuple_2d_m << ") => "
            << inner_product(tuple_2d_m, tuple_2d_m));
  ASSERT_TRUE(inner_product(tuple_2d_m, tuple_2d_m) == 80);

  // 3d s|d|m
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3d_s << ", " << tuple_3d_s << ") => "
            << inner_product(tuple_3d_s, tuple_3d_s));
  CUTE_STATIC_ASSERT_V(inner_product(tuple_3d_s, tuple_3d_s) == Int<84>{});
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3d_d << ", " << tuple_3d_d << ") => "
            << inner_product(tuple_3d_d, tuple_3d_d));
  ASSERT_TRUE(inner_product(tuple_3d_d, tuple_3d_d) == 84);
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3d_m << ", " << tuple_3d_m << ") => "
            << inner_product(tuple_3d_m, tuple_3d_m));
  ASSERT_TRUE(inner_product(tuple_3d_m, tuple_3d_m) == 84);

  // 3h s|d|m
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3h_s << ", " << tuple_3h_s << ") => "
            << inner_product(tuple_3h_s, tuple_3h_s));
  CUTE_STATIC_ASSERT_V(inner_product(tuple_3h_s, tuple_3h_s) == Int<73>{});
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3h_d << ", " << tuple_3h_d << ") => "
            << inner_product(tuple_3h_d, tuple_3h_d));
  ASSERT_TRUE(inner_product(tuple_3h_d, tuple_3h_d) == 73);
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3h_m << ", " << tuple_3h_m << ") => "
            << inner_product(tuple_3h_m, tuple_3h_m));
  ASSERT_TRUE(inner_product(tuple_3h_m, tuple_3h_m) == 73);

  CUTLASS_TRACE_HOST("col_major(" << tuple_2d_s << ") => " << compact_col_major(tuple_2d_s));
  CUTE_STATIC_ASSERT_V((compact_col_major(tuple_2d_s) == make_tuple(_1{},_8{})));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3d_s << ") => " << compact_col_major(tuple_3d_s));
  CUTE_STATIC_ASSERT_V((compact_col_major(tuple_3d_s) == make_tuple(_1{},_8{},_32{})));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3h_s << ") => " << compact_col_major(tuple_3h_s));
  CUTE_STATIC_ASSERT_V((compact_col_major(tuple_3h_s) == make_tuple(make_tuple(_0{},_1{}),_2{},_16{})));

  CUTLASS_TRACE_HOST("col_major(" << tuple_2d_d << ") => " << compact_col_major(tuple_2d_d));
  ASSERT_TRUE((compact_col_major(tuple_2d_d) == make_tuple(_1{},8)));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3d_d << ") => " << compact_col_major(tuple_3d_d));
  ASSERT_TRUE((compact_col_major(tuple_3d_d) == make_tuple(_1{},8,32)));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3h_d << ") => " << compact_col_major(tuple_3h_d));
  ASSERT_TRUE((compact_col_major(tuple_3h_d) == make_tuple(make_tuple(_1{},1),2,16)));

  CUTLASS_TRACE_HOST("col_major(" << tuple_2d_m << ") => " << compact_col_major(tuple_2d_m));
  ASSERT_TRUE((compact_col_major(tuple_2d_m) == make_tuple(_1{},_8{})));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3d_m << ") => " << compact_col_major(tuple_3d_m));
  ASSERT_TRUE((compact_col_major(tuple_3d_m) == make_tuple(_1{},8,32)));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3h_m << ") => " << compact_col_major(tuple_3h_m));
  ASSERT_TRUE((compact_col_major(tuple_3h_m) == make_tuple(make_tuple(_1{},1),2,16)));

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("SLICING TUPLES");
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto a = Coord<_2,_3,_4,Coord<_5,_6>>{};

    CUTLASS_TRACE_HOST("a = " << a);

    CUTLASS_TRACE_HOST("a(1) = " << slice(1, a));

    CUTLASS_TRACE_HOST("a(_) = " << slice(_, a));

    CUTLASS_TRACE_HOST("a(_,1,_,_) = " << slice(make_coord(_,1,_,_), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(_,_)) = " << slice(make_coord(_,1,_,make_coord(_,_)), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(_,2)) = " << slice(make_coord(_,1,_,make_coord(_,2)), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(1,2)) = " << slice(make_coord(_,1,_,make_coord(1,2)), a));
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("DICING TUPLES");
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto a = Coord<_2,_3,_4,Coord<_5,_6>>{};

    CUTLASS_TRACE_HOST("a = " << a);

    CUTLASS_TRACE_HOST("a(1) = " << dice(1, a));

    CUTLASS_TRACE_HOST("a(_) = " << dice(_, a));

    CUTLASS_TRACE_HOST("a(_,1,_,_) = " << dice(make_coord(_,1,_,_), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(_,_)) = " << dice(make_coord(_,1,_,make_coord(_,_)), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(_,2)) = " << dice(make_coord(_,1,_,make_coord(_,2)), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(1,2)) = " << dice(make_coord(_,1,_,make_coord(1,2)), a));
  }
}

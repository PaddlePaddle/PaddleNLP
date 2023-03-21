/*  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
    Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
    This code is copied fron NVIDIA apex: 
    https://github.com/NVIDIA/apex with minor changes.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and 
    limitations under the License. */

#pragma once

#include "paddle/extension.h"

namespace phi {
namespace dtype {

class float16;
class bfloat16;
    
}  // namespace dtype
}  // namespace phi

template <paddle::DataType TYPE_ENUM>
struct TypeTrait;

#define DEFINE_TYPE_TRAIT(__cpp_type, __type_enum)             \
template <>                                                    \
struct TypeTrait<paddle::DataType::__type_enum> {              \
  using Type = __cpp_type;                                     \
  static constexpr auto kType = paddle::DataType::__type_enum; \
}

DEFINE_TYPE_TRAIT(float, FLOAT32);
DEFINE_TYPE_TRAIT(double, FLOAT64);
DEFINE_TYPE_TRAIT(phi::dtype::float16, FLOAT16);
DEFINE_TYPE_TRAIT(phi::dtype::bfloat16, BFLOAT16);

#define CASE_IMPL(TYPE, ...)                                       \
     case paddle::DataType::TYPE: do {                             \
         using scalar_t = TypeTrait<paddle::DataType::TYPE>::Type; \
         __VA_ARGS__;                                              \
         break;                                                    \
     } while (0);                                                  \
     break  

#define DEFAULT_THROW(NAME, TYPE)                           \
    default: do {                                           \
      PD_THROW(#NAME, " not implemented for '", TYPE, "'"); \
    } while (0);                                            \
    break


#define DISPATCH_HALF_AND_BFLOAT(TYPE, NAME, ...) \
  switch(TYPE)					                  \
  {									              \
    CASE_IMPL(FLOAT16, __VA_ARGS__);              \
    CASE_IMPL(BFLOAT16, __VA_ARGS__);             \
    DEFAULT_THROW(NAME, TYPE);                    \
  }


#define DISPATCH_HALF_BFLOAT_AND_FLOAT(TYPE, NAME, ...)	\
  switch(TYPE)                                          \
  {                                                     \
    CASE_IMPL(FLOAT16, __VA_ARGS__);                    \
    CASE_IMPL(BFLOAT16, __VA_ARGS__);                   \
    CASE_IMPL(FLOAT32, __VA_ARGS__);                    \
    DEFAULT_THROW(NAME, TYPE);                          \
  }

#define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch(TYPEIN)							\
    {									\
    case paddle::DataType::FLOAT32:						\
      {									\
	using scalar_t_in = float;					\
	switch(TYPEOUT)							\
	  {								\
	  case paddle::DataType::FLOAT32:					\
	    {								\
	      using scalar_t_out = float;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case paddle::DataType::FLOAT16:					\
	    {								\
	      using scalar_t_out = phi::dtype::float16;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case paddle::DataType::BFLOAT16:				\
	    {								\
	      using scalar_t_out = phi::dtype::bfloat16;			\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  DEFAULT_THROW(NAME, TYPEOUT); \
	  }								\
	break;								\
      }									\
    case paddle::DataType::FLOAT16:						\
      {									\
	using scalar_t_in = phi::dtype::float16;					\
	using scalar_t_out = phi::dtype::float16;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    case paddle::DataType::BFLOAT16:					\
      {									\
	using scalar_t_in = phi::dtype::bfloat16;				\
	using scalar_t_out = phi::dtype::bfloat16;				\
	__VA_ARGS__;							\
	break;								\
      }									\
    DEFAULT_THROW(NAME, TYPEIN); \
    }



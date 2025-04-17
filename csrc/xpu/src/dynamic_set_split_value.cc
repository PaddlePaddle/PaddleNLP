// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <cassert> 
#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"
#include <xft/xdnn_plugin.h>
namespace xftkernel = baidu::xpu::xftkernel;
std::vector<paddle::Tensor> DynamicSetValue(const paddle::Tensor& input,
                                            const paddle::Tensor& index,
                                            paddle::Tensor* output,
                                            bool is_before_index) { // true: [ : index, :] ; false: [index : , :]

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  using XPUType = typename XPUTypeTrait<bfloat16>::Type; // only support bfloat16
  typedef paddle::bfloat16 data_t;

  int index_num = index.data<int64_t>()[0];
  int total_len = output->dims()[0];
  int dim = output->dims()[1];
  int copy_num = input.numel();
  assert(dim == input.dims()[1] && "input.dim != output.dim! ");
  assert((index_num >= 0 && index_num <= total_len) && "index out of range! ");
  if (is_before_index) {
    assert((index_num - 0) * dim == copy_num && "copy_num wrong! ");
  }
  else{
    assert((total_len - index_num) * dim == copy_num && "copy_num wrong! ");
  }
  int r = baidu::xpu::api::copy<XPUType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const XPUType*>(input.data<data_t>()),
      reinterpret_cast<XPUType*>(output->data<data_t>() + (is_before_index ? 0 : index_num * dim)),
      copy_num);

  return {};
}

PD_BUILD_OP(dynamic_set_value)
    .Inputs({"input", "index", "output"})
    .Outputs({"output_"})
    .SetInplaceMap({{"output_", "output"}})
    .Attrs({"is_before_index: bool"})
    .SetKernelFn(PD_KERNEL(DynamicSetValue));



// std::vector<paddle::Tensor> DynamicSplitValue(const paddle::Tensor& input,
//                                             const paddle::Tensor& index,
//                                             bool is_before_index) { // true: [ : index, :] ; false: [index : , :]

//   phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
//   auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
//   auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
//   using XPUType = typename XPUTypeTrait<bfloat16>::Type; // only support bfloat16
//   typedef paddle::bfloat16 data_t;



  
//   int index_num = index.data<int64_t>()[0];
//   int total_len = input.dims()[0];
//   int dim = input.dims()[1];
//   int copy_len = 0;

//   if (is_before_index) {
//     copy_len = index_num
//   }
//   else{
//     copy_len = total_len - index_num
//   }  

//   auto output = paddle::full({copy_len, dim}, 0, seq_lens_encoder.type(), seq_lens_encoder.place());
//   assert(dim == input.dims()[1] && "input.dim != output.dim! ");
//   assert((index_num >= 0 && index_num <= total_len) && "index out of range! ");
//   if (is_before_index) {
//     assert((index_num - 0) * dim == copy_len && "copy_len wrong! ");
//   }
//   else{
//     assert((total_len - index_num) * dim == copy_len && "copy_len wrong! ");
//   }
//   int r = baidu::xpu::api::copy<XPUType>(
//       xpu_ctx->x_context(),
//       reinterpret_cast<const XPUType*>(input.data<data_t>()),
//       reinterpret_cast<XPUType*>(output->data<data_t>() + (is_before_index ? 0 : index_num * dim)),
//       copy_len);

//   return {};
// }

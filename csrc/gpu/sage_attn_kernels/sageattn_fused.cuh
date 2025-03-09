#pragma once
#include "paddle/extension.h"

std::vector<paddle::Tensor> per_warp_int8_cuda(paddle::Tensor& q,
                                            paddle::Tensor& k,
                                            paddle::Tensor& km,
                                            int BLKQ,
                                            int WARPQ,
                                            int BLKK,
                                            int tensor_layout);
                    
std::vector<paddle::Tensor> per_channel_fp8(paddle::Tensor& v,
                                            int tensor_layout,
                                            float scale_max,
                                            bool smooth_v);

std::vector<paddle::Tensor> sub_mean(paddle::Tensor& v,
                                    paddle::Tensor& vm,
                                    int tensor_layout);
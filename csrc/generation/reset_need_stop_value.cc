#include "paddle/extension.h"

void SetStopValue(const paddle::Tensor& not_need_stop) {
  bool *stop_data = const_cast<bool*>(not_need_stop.data<bool>());
  stop_data[0] = true;
}

PD_BUILD_OP(reset_stop_value)
    .Inputs({"not_need_stop"})
    .Outputs({"not_need_stop_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"}})
    .SetKernelFn(PD_KERNEL(SetStopValue));

#include "sm.h"


int GetSMVersion() {
  int device{-1};
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  return props.major * 10 + props.minor;
}

SMVersion* SMVersion::GetInstance() {
  static SMVersion* sm_version_ = nullptr;
  if (sm_version_ == nullptr) {
    sm_version_ = new SMVersion();
  }
  return sm_version_;
}

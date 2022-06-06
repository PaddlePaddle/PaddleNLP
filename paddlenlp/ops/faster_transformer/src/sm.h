#pragma once

#include <cuda_runtime.h>

int GetSMVersion();

class SMVersion {
private:
  SMVersion() { sm_ = GetSMVersion(); }

public:
  SMVersion(SMVersion& other) = delete;

  void operator=(const SMVersion&) = delete;

  static SMVersion* GetInstance();

  ~SMVersion();

  int sm_;
};

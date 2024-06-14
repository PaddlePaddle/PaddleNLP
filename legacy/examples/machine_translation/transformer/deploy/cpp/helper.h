#pragma once
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <glog/raw_logging.h>
#include <sys/time.h>
#include <chrono>  // NOLINT
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/include/paddle_inference_api.h"

namespace paddle {
namespace inference {
// Timer for timer
class Timer {
public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;
  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

static void split(const std::string &str,
                  char sep,
                  std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

}  // namespace inference
}  // namespace paddle

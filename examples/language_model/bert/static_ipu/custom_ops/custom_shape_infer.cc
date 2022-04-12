/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <popart/shapeinference.hpp>
#include <popart/version.hpp>

auto splitShapeInferenceFun = [](popart::ShapeInferenceContext &ctx) {
  auto numOutputs = ctx.getNumOutputs();
  auto type = ctx.inType(0);
  auto shape = ctx.inShape(0);
  auto axis = ctx.getAttribute<int64_t>("axis");
  auto split = ctx.getAttribute<std::vector<int64_t>>("split");

  for (int i = 0; i < numOutputs; i++) {
    shape[axis] = split.at(i);
    ctx.outInfo(i) = {type, shape};
  }
};

#if POPART_VERSION_MAJOR == 2
#if POPART_VERSION_MINOR == 3
// for version 2.3, need to register a shape inference function for Split op
static popart::RegisterShapeInferenceFunction
    splitRegister11(popart::Onnx::Operators::Split_11, splitShapeInferenceFun);
#endif
#endif
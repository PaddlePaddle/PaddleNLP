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

#include "fast_tokenizer/postprocessors/postprocessor.h"
#include "fast_tokenizer/core/encoding.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace postprocessors {

void PostProcessor::DefaultProcess(core::Encoding* encoding,
                                   core::Encoding* pair_encoding,
                                   core::Encoding* result_encoding) {
  if (pair_encoding == nullptr) {
    *result_encoding = *encoding;
  } else {
    encoding->SetSequenceIds(0);
    pair_encoding->SetSequenceIds(1);
    encoding->MergeWith(*pair_encoding, false);
    *result_encoding = *encoding;
  }
}

}  // namespace postprocessors
}  // namespace fast_tokenizer
}  // namespace paddlenlp

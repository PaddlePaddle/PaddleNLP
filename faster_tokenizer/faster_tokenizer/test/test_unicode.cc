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

#include <string>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "normalizers/bert.h"
#include "normalizers/replace.h"
#include "normalizers/strip.h"
#include "normalizers/unicode.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace tests {

TEST(normalizers, unicode) {
  std::string input = "\u1e9b\u0323a\u1e9b\u0323";
  std::string expected_nfkc_output = "ṩaṩ";
  std::string expected_nfc_output = "\u1e9b\u0323a\u1e9b\u0323";
  std::string expected_nfkd_output = "\u0073\u0323\u0307a\u0073\u0323\u0307";
  std::string expected_nfd_output = "\u017f\u0323\u0307a\u017f\u0323\u0307";

  normalizers::NFKCNormalizer nfkc;
  normalizers::NormalizedString normalized_input1(input);
  normalizers::NormalizedString normalized_input2(input);
  normalizers::NormalizedString normalized_input3(input);
  normalizers::NormalizedString normalized_input4(input);
  nfkc(&normalized_input1);
  std::string nfkc_output = normalized_input1.GetStr();
  ASSERT_EQ(expected_nfkc_output, nfkc_output);

  normalizers::NFCNormalizer nfc;
  nfc(&normalized_input2);
  std::string nfc_output = normalized_input2.GetStr();
  ASSERT_EQ(expected_nfc_output, nfc_output);

  normalizers::NFKDNormalizer nfkd;
  nfkd(&normalized_input3);
  std::string nfkd_output = normalized_input3.GetStr();
  ASSERT_EQ(expected_nfkd_output, nfkd_output);

  normalizers::NFDNormalizer nfd;
  nfd(&normalized_input4);
  std::string nfd_output = normalized_input4.GetStr();
  ASSERT_EQ(expected_nfd_output, nfd_output);
}

}  // namespace tests
}  // namespace faster_tokenizer
}  // namespace paddlenlp
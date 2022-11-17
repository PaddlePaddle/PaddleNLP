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

#include <iostream>
#include <vector>
#include "fast_tokenizer/tokenizers/ernie_fast_tokenizer.h"
using namespace paddlenlp;

int main() {
  // 1. Define a ernie fast tokenizer
  fast_tokenizer::tokenizers_impl::ErnieFastTokenizer tokenizer(
      "ernie_vocab.txt");
  // 2. Tokenize the input strings
  // case 1: tokenize a single string
  std::cout << "case 1: Tokenize a single string" << std::endl;
  fast_tokenizer::core::Encoding encoding;
  std::string single_string =
      "商赢环球股份有限公司关于延期回复上海证券交易所对"
      "公司2017年年度报告的事后审核问询函的公告";
  tokenizer.EncodePairStrings(single_string, &encoding);
  std::cout << encoding.DebugString() << std::endl;

  // case 2: tokenize a pair of strings
  std::cout << "case 2: Tokenize a pair of strings" << std::endl;
  std::string text = "蚂蚁借呗等额还款可以换成先息后本吗";
  std::string text_pair = "借呗有先息到期还本吗";

  tokenizer.EncodePairStrings(text, text_pair, &encoding);
  std::cout << encoding.DebugString() << std::endl;

  // case 3: Tokenize a batch of single strings
  std::cout << "case 3: Tokenize a batch of single strings" << std::endl;
  std::vector<fast_tokenizer::core::Encoding> encodings;
  std::vector<std::string> strings_list = {
      "通过中介公司买了二手房，首付都付了，现在卖家不想卖了。怎么处理？",
      "凌云研发的国产两轮电动车怎么样，有什么惊喜？",
      "一辆车的寿命到底多长，最多可以开多久？"};
  tokenizer.EncodeBatchStrings(strings_list, &encodings);
  for (auto&& encoding : encodings) {
    std::cout << encoding.DebugString() << std::endl;
  }

  // case 4: Tokenize a batch of pair strings
  std::cout << "case 4: Tokenize a batch of pair strings" << std::endl;
  std::vector<std::string> texts = {
      "花呗自动从余额宝扣款，需要我自己设置吗",
      "这个蚂蚁花呗能恢复正常用不",
      "在经济的另一次转变中，人们发现在低地农场饲养羔羊更具成本效益，部分原因"
      "是那里有更丰富、更有营养的牧场，因此湖地农场的利润变得更少。"};
  std::vector<std::string> text_pairs = {
      "支付宝余额会自动还花呗吗",
      "我的蚂蚁花呗 怎么用不了",
      "人们发现，经济的另一个转变更有营养。"};
  tokenizer.EncodeBatchStrings(texts, text_pairs, &encodings);
  for (auto&& encoding : encodings) {
    std::cout << encoding.DebugString() << std::endl;
  }
  return 0;
}
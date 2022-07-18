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

#include <sstream>

#include "core/added_vocabulary.h"
#include "core/base.h"
#include "core/encoding.h"
#include "pybind/core.h"

#include <Python.h>
#include <pybind11/operators.h>

namespace py = pybind11;

namespace paddlenlp {
namespace faster_tokenizer {
namespace pybind {

py::list GetWordIdx(const core::Encoding& self) {
  py::list list;
  for (const auto& idx : self.GetWordsIdx()) {
    if (idx == static_cast<uint32_t>(-1)) {
      list.append(py::none());
    } else {
      list.append(py::cast(idx));
    }
  }
  return list;
}

void BindCore(pybind11::module* m) {
  py::class_<core::Token>(*m, "Token")
      .def(py::init<>())
      .def_readwrite("id", &core::Token::id_)
      .def_readwrite("value", &core::Token::value_)
      .def_readwrite("offset", &core::Token::offset_)
      .def("__repr__", [](const core::Token& token) {
        std::ostringstream oss;
        oss << "id: " << token.id_ << "\tvalue:" << token.value_
            << "\toffset: (" << token.offset_.first << ", "
            << token.offset_.second << ")";
        return oss.str();
      });
  py::class_<core::PadMethod>(*m, "PadMethod")
      .def(py::init<>())
      .def_readwrite("strategy", &core::PadMethod::strategy_)
      .def_readwrite("direction", &core::PadMethod::direction_)
      .def_readwrite("pad_id", &core::PadMethod::pad_id_)
      .def_readwrite("pad_token_type_id", &core::PadMethod::pad_token_type_id_)
      .def_readwrite("pad_token", &core::PadMethod::pad_token_)
      .def_readwrite("pad_len", &core::PadMethod::pad_len_)
      .def_readwrite("pad_to_multiple_of",
                     &core::PadMethod::pad_to_multiple_of_);
  py::class_<core::TruncMethod>(*m, "TruncMethod")
      .def(py::init<>())
      .def_readwrite("direction", &core::TruncMethod::direction_)
      .def_readwrite("max_len", &core::TruncMethod::max_len_)
      .def_readwrite("strategy", &core::TruncMethod::strategy_)
      .def_readwrite("stride", &core::TruncMethod::stride_);

  py::enum_<core::OffsetType>(*m, "OffsetType")
      .value("CHAR", core::OffsetType::CHAR)
      .value("BYTE", core::OffsetType::BYTE)
      .export_values();
  py::enum_<core::Direction>(*m, "Direction")
      .value("LEFT", core::Direction::LEFT)
      .value("RIGHT", core::Direction::RIGHT)
      .export_values();
  py::enum_<core::TruncStrategy>(*m, "TruncStrategy")
      .value("LONGEST_FIRST", core::TruncStrategy::LONGEST_FIRST)
      .value("ONLY_FIRST", core::TruncStrategy::ONLY_FIRST)
      .value("ONLY_SECOND", core::TruncStrategy::ONLY_SECOND)
      .export_values();
  py::enum_<core::PadStrategy>(*m, "PadStrategy")
      .value("BATCH_LONGEST", core::PadStrategy::BATCH_LONGEST)
      .value("FIXED_SIZE", core::PadStrategy::FIXED_SIZE)
      .export_values();

  py::class_<core::Encoding>(*m, "Encoding")
      .def(py::init<const std::vector<uint32_t>&,
                    const std::vector<uint32_t>&,
                    const std::vector<std::string>&,
                    const std::vector<uint32_t>&,
                    const std::vector<core::Offset>&,
                    const std::vector<uint32_t>&,
                    const std::vector<uint32_t>&,
                    const std::vector<core::Encoding>&,
                    const std::unordered_map<uint32_t, core::Range>&>(),
           py::arg("ids"),
           py::arg("type_ids"),
           py::arg("tokens"),
           py::arg("words_idx"),
           py::arg("offsets"),
           py::arg("special_tokens_mask"),
           py::arg("attention_mask"),
           py::arg("overflowing"),
           py::arg("sequence_ranges"))
      .def(py::init<uint32_t>(), py::arg("size"))
      .def(py::init<const std::vector<core::Token>&, uint32_t>(),
           py::arg("tokens"),
           py::arg("type_id"))
      .def("__str__", &core::Encoding::DebugString)
      .def("__repr__", &core::Encoding::DebugString)
      .def("__len__", &core::Encoding::GetLen)
      .def_property_readonly("n_sequences", &core::Encoding::GetNumSequence)
      .def_property_readonly("tokens", &core::Encoding::GetTokens)
      .def_property_readonly("word_ids", &GetWordIdx)
      .def_property_readonly("sequence_ids", &core::Encoding::GetSequenceIds)
      .def_property_readonly("ids", &core::Encoding::GetIds)
      .def_property_readonly("type_ids", &core::Encoding::GetTypeIds)
      .def_property_readonly("offsets", &core::Encoding::GetOffsets)
      .def_property_readonly("special_tokens_mask",
                             &core::Encoding::GetSpecialTokensMask)
      .def_property_readonly("attention_mask",
                             &core::Encoding::GetAttentionMask)
      .def_property_readonly("overflowing", &core::Encoding::GetOverflowing)
      .def("set_sequence_ids",
           &core::Encoding::SetSequenceIds,
           py::arg("sequence_id"))
      .def("char_to_token",
           [](const core::Encoding& self,
              uint32_t char_pos,
              uint32_t seq_id) -> py::object {
             auto token_idxs = self.CharOffsetsToTokenIdx(char_pos, seq_id);
             if (token_idxs.size() == 0) {
               return py::none();
             }
             return py::cast(token_idxs[0]);
           },
           py::arg("char_pos"),
           py::arg("sequence_index") = 0)
      .def("char_to_word",
           [](const core::Encoding& self,
              uint32_t char_pos,
              uint32_t seq_id) -> py::object {
             auto word_idxs = self.CharOffsetsToWordIdx(char_pos, seq_id);
             if (word_idxs.size() == 0) {
               return py::none();
             }
             return py::cast(word_idxs[0]);
           },
           py::arg("char_pos"),
           py::arg("sequence_index") = 0)
      .def_static("merge",
                  &core::Encoding::Merge,
                  py::arg("encodings"),
                  py::arg("growing_offsets") = true)
      .def("pad",
           [](core::Encoding& self,
              uint32_t length,
              const std::string& direction,
              uint32_t pad_id,
              uint32_t pad_type_id,
              const std::string& pad_token) {
             core::Direction direct;
             if (direction == "right") {
               direct = core::Direction::RIGHT;
             } else {
               direct = core::Direction::LEFT;
             }
             self.Pad(length, pad_id, pad_type_id, pad_token, direct);
           },
           py::arg("length"),
           py::arg("direction") = "right",
           py::arg("pad_id") = 0,
           py::arg("pad_type_id") = 0,
           py::arg("pad_token") = "[PAD]")
      .def("token_to_chars",
           [](const core::Encoding& self, uint32_t token_index) -> py::object {
             auto offsets = self.TokenIdxToCharOffsets(token_index);
             if (offsets.size() == 0) {
               return py::none();
             }
             return py::cast(offsets[0]);
           },
           py::arg("token_index"))
      .def("token_to_sequence",
           [](const core::Encoding& self, uint32_t token_index) -> py::object {
             auto seq_ids = self.TokenIdxToSequenceIds(token_index);
             if (seq_ids.size() == 0) {
               return py::none();
             }
             return py::cast(seq_ids[0]);
           },
           py::arg("token_index"))
      .def("token_to_word",
           [](const core::Encoding& self, uint32_t token_index) -> py::object {
             auto word_idx = self.TokenIdxToWordIdx(token_index);
             if (word_idx.size() == 0) {
               return py::none();
             }
             return py::cast(word_idx[0].second);
           },
           py::arg("token_index"))
      .def("word_to_chars",
           [](const core::Encoding& self,
              uint32_t word_index,
              uint32_t sequence_index) -> py::object {
             auto ranges =
                 self.WordIdxToCharOffsets(word_index, sequence_index);
             if (ranges.size() == 0) {
               return py::none();
             }
             return py::cast(ranges[0]);
           },
           py::arg("word_index"),
           py::arg("sequence_index") = 0)
      .def("word_to_tokens",
           [](const core::Encoding& self,
              uint32_t word_index,
              uint32_t sequence_index) -> py::object {
             auto ranges = self.WordIdxToTokensIdx(word_index, sequence_index);
             if (ranges.size() == 0) {
               return py::none();
             }
             return py::cast(ranges[0]);
           },
           py::arg("word_index"),
           py::arg("sequence_index") = 0)
      .def("truncate",
           [](core::Encoding& self,
              size_t max_length,
              size_t stride,
              const std::string& direction) {
             core::Direction direct;
             if (direction == "right") {
               direct = core::Direction::RIGHT;
             } else {
               direct = core::Direction::LEFT;
             }
             self.Truncate(max_length, stride, direct);
           },
           py::arg("max_length"),
           py::arg("stride") = 0,
           py::arg("direction") = "right");

  py::class_<core::AddedToken>(*m, "AddedToken")
      .def(py::init<>())
      .def(py::init([](const std::string& content,
                       bool single_word,
                       bool lstrip,
                       bool rstrip,
                       bool normalized) {
             return core::AddedToken(
                 content, !normalized, single_word, lstrip, rstrip);
           }),
           py::arg("content"),
           py::arg("single_word") = false,
           py::arg("lstrip") = false,
           py::arg("rstrip") = false,
           py::arg("normalized") = true)
      .def(py::self == py::self)
      .def_property_readonly("content", &core::AddedToken::GetContent)
      .def_property_readonly("get_is_special", &core::AddedToken::GetIsSpecial)
      .def_property_readonly(
          "normalized",
          [](const core::AddedToken& self) { return !self.GetUseNormalized(); })
      .def_property_readonly("lstrip", &core::AddedToken::GetUseLStrip)
      .def_property_readonly("rstrip", &core::AddedToken::GetUseRStrip)
      .def_property_readonly("single_word", &core::AddedToken::GetIsSingleWord);
}

}  // namespace pybind
}  // namespace faster_tokenizer
}  // namespace paddlenlp

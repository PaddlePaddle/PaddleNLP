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

#pragma once

#include <functional>
#include <string>
#include <vector>
#include "core/base.h"
#include "glog/logging.h"

namespace re2 {
class RE2;
}  // re2

namespace tokenizers {
namespace normalizers {

enum SplitMode {
  REMOVED,
  ISOLATED,
  MERGED_WITH_PREVIOUS,
  MERGED_WITH_NEXT,
  CONTIGUOUS
};

struct OffsetMapping {
  std::u32string u32normalized;
  std::vector<int> changes;  // Same size as normalized
};

class NormalizedString {
public:
  NormalizedString(const std::string& original);
  NormalizedString(NormalizedString&& other);
  NormalizedString(const NormalizedString& other) = default;
  NormalizedString& operator=(const NormalizedString& other) = default;
  NormalizedString& operator=(NormalizedString&& other);
  const std::string& GetStr() const;
  const std::string& GetOrignalStr() const;
  uint GetLen() const;
  uint GetOriginalLen() const;
  core::Offset GetOrginalOffset() const;
  bool IsEmpty() const;
  bool IsOriginalEmpty() const;

  // Unicode Normalization
  NormalizedString& NFD();
  NormalizedString& NFKD();
  NormalizedString& NFC();
  NormalizedString& NFKC();

  // Strip
  NormalizedString& LRStrip(bool left, bool right);
  NormalizedString& LStrip();
  NormalizedString& RStrip();

  NormalizedString& FilterChar(std::function<bool(char32_t)> keep_char_fn);
  NormalizedString& MapChar(std::function<char32_t(char32_t)> map_char_fn);
  NormalizedString& Lowercase();
  NormalizedString& Replace(const re2::RE2& pattern,
                            const std::string& content);
  bool Slice(core::Range range,
             NormalizedString* normalized,
             bool origin_range) const;

  void UpdateNormalized(const OffsetMapping& new_normalized,
                        uint initial_offset);
  template <typename PatternType>
  void Split(const PatternType&
                 pattern, /* re2::RE2 or std::function<bool(char32_t)> */
             SplitMode mode,
             std::vector<NormalizedString>* normalizes) const {
    // Vec<(Offsets, should_remove)>
    std::vector<std::pair<core::Range, bool>> matches;
    auto normalizes_size = GetMatch(normalized_, pattern, &matches);
    // Convert matches
    switch (mode) {
      case REMOVED:
        break;
      case ISOLATED: {
        for (auto& match : matches) {
          match.second = false;
        }
        normalizes_size = matches.size();
        break;
      }
      case MERGED_WITH_PREVIOUS: {
        bool previous_match = false;
        std::vector<std::pair<core::Range, bool>> new_matches;
        for (const auto& match : matches) {
          auto offset = match.first;
          bool curr_match = match.second;
          if (curr_match && !previous_match) {
            if (new_matches.size() > 0) {
              new_matches.back().first.second = offset.second;
            } else {
              new_matches.push_back({offset, false});
            }
          } else {
            new_matches.push_back({offset, false});
          }
          previous_match = curr_match;
        }
        matches = std::move(new_matches);
        normalizes_size = matches.size();
        break;
      }
      case MERGED_WITH_NEXT: {
        bool previous_match = false;
        std::vector<std::pair<core::Range, bool>> new_matches;
        for (auto it = matches.crbegin(); it != matches.crend(); ++it) {
          const auto& match = *it;
          auto offset = match.first;
          bool curr_match = match.second;
          if (curr_match && !previous_match) {
            if (new_matches.size() > 0) {
              new_matches.back().first.first = offset.first;
            } else {
              new_matches.push_back({offset, false});
            }
          } else {
            new_matches.push_back({offset, false});
          }
          previous_match = curr_match;
        }
        matches = std::move(new_matches);
        int end = matches.size();
        normalizes_size = matches.size();
        for (int i = 0; i < end / 2; ++i) {
          std::swap(matches[i], matches[end - i - 1]);
        }
        break;
      }
      case CONTIGUOUS: {
        bool previous_match = false;
        std::vector<std::pair<core::Range, bool>> new_matches;
        for (const auto& match : matches) {
          auto offset = match.first;
          bool curr_match = match.second;
          if (curr_match == previous_match) {
            if (new_matches.size() > 0) {
              new_matches.back().first.second = offset.second;
            } else {
              new_matches.push_back({offset, false});
            }
          } else {
            new_matches.push_back({offset, false});
          }
          previous_match = curr_match;
        }
        matches = std::move(new_matches);
        normalizes_size = matches.size();
        break;
      }
      default:
        break;
    }
    normalizes->resize(normalizes_size);
    int idx = 0;
    for (const auto& match : matches) {
      if (!match.second) {
        Slice(match.first, &(normalizes->at(idx++)), false);
      }
    }
  }
  bool ConvertOffsets(core::Range* range, bool origin_range = true) const;
  NormalizedString() = default;

private:
  std::string original_;
  std::string normalized_;
  // In order to keep track of the offset mapping from
  // original_ to normalized_
  std::vector<core::Range> alignments_;
  uint original_shift_;

  void UpdateNormalizedRange(const OffsetMapping& new_normalized,
                             uint initial_offset,
                             const core::Range& range,
                             bool origin_range = true);
  void RunNormalization(const std::string& mode);
  bool ValidateRange(const core::Range& range, bool origin_range) const;

  uint32_t GetMatch(const std::string& normalized,
                    const re2::RE2& pattern,
                    std::vector<std::pair<core::Range, bool>>* matches) const;

  uint32_t GetMatch(const std::string& normalized,
                    const std::function<bool(char32_t)>& pattern_func,
                    std::vector<std::pair<core::Range, bool>>* matches) const;
};

struct Normalizer {
  virtual void operator()(NormalizedString* mut_str) const = 0;
};

}  // normalizers
}  // tokenizers

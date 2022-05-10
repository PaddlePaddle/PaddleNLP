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
#include <unordered_map>
#include <vector>
#include "core/base.h"

namespace tokenizers {
namespace core {

class Encoding {
public:
  Encoding() = default;
  Encoding(const std::vector<uint>& ids,
           const std::vector<uint>& type_ids,
           const std::vector<std::string>& tokens,
           const std::vector<uint>& words_idx,
           const std::vector<Offset>& offsets,
           const std::vector<uint>& special_tokens_mask,
           const std::vector<uint>& attention_mask,
           const std::vector<Encoding>& overflowing,
           const std::unordered_map<uint, Range>& sequence_ranges);
  // Move version
  Encoding(std::vector<uint>&& ids,
           std::vector<uint>&& type_ids,
           std::vector<std::string>&& tokens,
           std::vector<uint>&& words_idx,
           std::vector<Offset>&& offsets,
           std::vector<uint>&& special_tokens_mask,
           std::vector<uint>&& attention_mask,
           std::vector<Encoding>&& overflowing,
           std::unordered_map<uint, Range>&& sequence_ranges);
  Encoding(uint size);
  Encoding(const std::vector<Token>& tokens, uint type_id);

  Encoding(Encoding&&);
  Encoding(const Encoding&) = default;
  Encoding& operator=(Encoding&&);
  Encoding& operator=(const Encoding&) = default;

  bool IsEmpty() const;
  void SetSequenceIds(uint seq_ids);

  // Getter
  int GetLen() const;
  int GetNumSequence() const;
  const std::vector<std::string>& GetTokens() const;
  const std::vector<uint>& GetWordsIdx() const;
  std::vector<uint>& GetMutableWordsIdx();
  std::vector<uint> GetSequenceIds() const;
  const std::vector<uint>& GetIds() const;
  const std::vector<uint>& GetTypeIds() const;
  const std::vector<Offset>& GetOffsets() const;
  std::vector<Offset>& GetMutableOffsets();
  const std::vector<uint>& GetSpecialTokensMask() const;
  const std::vector<uint>& GetAttentionMask() const;
  const std::vector<Encoding>& GetOverflowing() const;
  std::vector<Encoding>& GetMutableOverflowing();
  Range GetSequenceRange(uint seq_id) const;

  void ProcessTokenWithOffsets(
      std::function<void(uint, std::string*, Offset*)> process_token_fn);

  // token_idx: The index of token in the sequence
  std::vector<uint> TokenIdxToSequenceIds(uint token_idx) const;
  std::vector<Range> WordIdxToTokensIdx(uint word_idx, uint seq_id) const;
  std::vector<Offset> WordIdxToCharOffsets(uint word_idx, uint seq_id) const;
  std::vector<std::pair<uint, Offset>> TokenIdxToCharOffsets(
      uint token_idx) const;
  std::vector<std::pair<uint, uint>> TokenIdxToWordIdx(uint token_idx) const;
  std::vector<uint> CharOffsetsToTokenIdx(uint char_pos, uint seq_id) const;
  std::vector<uint> CharOffsetsToWordIdx(uint char_pos, uint seq_id) const;
  void Truncate(size_t max_len, size_t stride, Direction direction);
  void MergeWith(const Encoding& pair, bool growing_offsets);
  void Pad(uint target_length,
           uint pad_id,
           uint pad_type_id,
           const std::string& pad_token,
           Direction direction);
  // Static method
  static Encoding Merge(const std::vector<Encoding>& encodings,
                        bool growing_offsets);
  std::string DebugString() const;

private:
  std::vector<uint> ids_;
  std::vector<uint> type_ids_;
  std::vector<std::string> tokens_;
  std::vector<uint> words_idx_;
  std::vector<Offset> offsets_;
  std::vector<uint> special_tokens_mask_;
  std::vector<uint> attention_mask_;
  std::vector<Encoding> overflowing_;
  std::unordered_map<uint, Range> sequence_ranges_;
};

bool TruncateEncodings(Encoding* encoding,
                       Encoding* pair_encoding,
                       const TruncMethod& method);
void PadEncodings(std::vector<Encoding>* encoding, const PadMethod& method);

}  // core
}  // tokenizers

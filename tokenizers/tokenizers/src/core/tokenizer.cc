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

#include "glog/logging.h"

#include "core/added_vocabulary.h"
#include "core/base.h"
#include "core/encoding.h"
#include "core/tokenizer.h"

#include "models/models.h"
#include "normalizers/normalizers.h"
#include "postprocessors/postprocessors.h"
#include "pretokenizers/pretokenizers.h"

namespace tokenizers {
namespace core {

normalizers::Normalizer* Tokenizer::GetNormalizerPtr() const {
  return normalizer_.get();
}

void Tokenizer::ReleaseNormaizer() { normalizer_ = nullptr; }

pretokenizers::PreTokenizer* Tokenizer::GetPreTokenizer() const {
  return pretokenizer_.get();
}

void Tokenizer::ReleasePreTokenizer() { pretokenizer_ = nullptr; }

void Tokenizer::SetTruncMethod(const TruncMethod& trunc_method) {
  trunc_method_ = trunc_method;
}

void Tokenizer::EnableTruncMethod(size_t max_len,
                                  size_t stride,
                                  Direction direction,
                                  TruncStrategy strategy) {
  use_truncation_ = true;
  trunc_method_.direction_ = direction;
  trunc_method_.max_len_ = max_len;
  trunc_method_.strategy_ = strategy;
  trunc_method_.stride_ = stride;
}

void Tokenizer::DisableTruncMethod() { use_truncation_ = false; }

TruncMethod Tokenizer::GetTruncMethod() const { return trunc_method_; }

void Tokenizer::SetPadMethod(const PadMethod& pad_method) {
  pad_method_ = pad_method;
}

void Tokenizer::EnablePadMethod(Direction direction,
                                uint pad_id,
                                uint pad_type_id,
                                const std::string& pad_token,
                                uint* length,
                                uint* pad_to_multiple_of) {
  use_padding_ = true;
  pad_method_.direction_ = direction;
  pad_method_.pad_id_ = pad_id;
  pad_method_.pad_token_type_id_ = pad_type_id;
  pad_method_.pad_token_ = pad_token;
  if (length != nullptr) {
    pad_method_.pad_len_ = *length;
    pad_method_.strategy_ = PadStrategy::FIXED_SIZE;
  } else {
    pad_method_.strategy_ = PadStrategy::BATCH_LONGEST;
  }
  if (pad_to_multiple_of != nullptr) {
    pad_method_.pad_to_mutiple_of = *pad_to_multiple_of;
  }
}
void Tokenizer::DisablePadMethod() { use_padding_ = false; }

PadMethod Tokenizer::GetPadMethod() const { return pad_method_; }

models::Model* Tokenizer::GetModelPtr() const { return model_.get(); }

void Tokenizer::ReleasePostProcessor() { post_processor_ = nullptr; }

postprocessors::PostProcessor* Tokenizer::GetPostProcessorPtr() const {
  return post_processor_.get();
}

Vocab Tokenizer::GetVocab(bool with_added_vocabulary) const {
  auto vocab = model_->GetVocab();
  auto added_vocab = added_vocabulary_.GetVocab();
  if (with_added_vocabulary) {
    for (const auto& vocab_item : added_vocab) {
      vocab.insert(vocab_item);
    }
  }
  return vocab;
}

size_t Tokenizer::GetVocabSize(bool with_added_vocabulary) const {
  size_t vocab_size = model_->GetVocabSize();
  if (with_added_vocabulary) {
    vocab_size += added_vocabulary_.GetLen();
  }
  return vocab_size;
}

size_t Tokenizer::AddTokens(const std::vector<AddedToken>& tokens) {
  return added_vocabulary_.AddTokens(tokens, *model_, normalizer_.get());
}

size_t Tokenizer::AddSpecialTokens(const std::vector<AddedToken>& tokens) {
  return added_vocabulary_.AddSpecialTokens(tokens, *model_, normalizer_.get());
}

bool Tokenizer::TokenToId(const std::string& token, uint* id) const {
  return added_vocabulary_.TokenToId(token, *model_, id);
}

bool Tokenizer::IdToToken(uint id, std::string* token) const {
  return added_vocabulary_.IdToToken(id, *model_, token);
}

bool Tokenizer::DoTokenize(pretokenizers::PreTokenizedString* pretokenized,
                           uint type_id,
                           const std::vector<uint>& word_idx,
                           OffsetType offset_type,
                           Encoding* encoding) const {
  auto split_size = pretokenized->GetSplitsSize();
  pretokenized->Tokenize([&](normalizers::NormalizedString* normalized) {
    return this->GetModelPtr()->Tokenize(normalized->GetStr());
  });
  return pretokenized->TransformToEncoding(
      word_idx, type_id, offset_type, encoding);
}

bool Tokenizer::DoPreTokenize(
    pretokenizers::PreTokenizedString* pretokenized) const {
  if (pretokenizer_ != nullptr) {
    (*pretokenizer_)(pretokenized);
  }
  return true;
}

struct InputStringVisitor : public boost::static_visitor<> {
  InputStringVisitor(const Tokenizer* tokenizer,
                     uint type_id,
                     OffsetType offset_type,
                     Encoding* encodings)
      : tokenizer_(tokenizer),
        type_id_(type_id),
        offset_type_(offset_type),
        encodings_(encodings) {}
  void operator()(const std::vector<std::string>& pretokenized_texts) const {
    tokenizer_->EncodeSingleText(
        pretokenized_texts, type_id_, offset_type_, encodings_);
  }

  void operator()(const std::string& raw_text) const {
    tokenizer_->EncodeSingleText(raw_text, type_id_, offset_type_, encodings_);
  }
  const Tokenizer* tokenizer_;
  uint type_id_;
  OffsetType offset_type_;
  Encoding* encodings_;
};

void Tokenizer::EncodeSingleString(const InputString& input_string,
                                   uint type_id,
                                   OffsetType offset_type,
                                   Encoding* encodings) const {
  boost::apply_visitor(
      InputStringVisitor(this, type_id, offset_type, encodings), input_string);
}

void Tokenizer::PostProcess(Encoding* encoding,
                            Encoding* pair_encoding,
                            bool add_special_tokens,
                            Encoding* result_encoding) const {
  // 1. Trunc
  if (use_truncation_) {
    auto added_tokens_num = 0;
    if (post_processor_ != nullptr) {
      added_tokens_num =
          post_processor_->AddedTokensNum(pair_encoding != nullptr);
    }
    if (add_special_tokens && added_tokens_num > 0) {
      auto trunc_method = trunc_method_;
      trunc_method.max_len_ -= added_tokens_num;
      TruncateEncodings(encoding, pair_encoding, trunc_method);
    } else {
      TruncateEncodings(encoding, pair_encoding, trunc_method_);
    }
  }
  // 2. Post process
  if (post_processor_ == nullptr) {
    postprocessors::PostProcessor::DefaultProcess(
        encoding, pair_encoding, result_encoding);
  } else {
    (*post_processor_)(
        encoding, pair_encoding, add_special_tokens, result_encoding);
  }
  // 3. Pad
  if (use_padding_) {
    std::vector<Encoding> encodings;
    encodings.push_back(*result_encoding);
    PadEncodings(&encodings, pad_method_);
  }
}

void Tokenizer::EncodePairStrings(const EncodeInput& encode_input,
                                  bool add_special_tokens,
                                  Encoding* encodings) const {
  Encoding encoding;
  if (encode_input.type() == typeid(InputString)) {
    const auto& input_string = boost::get<InputString>(encode_input);
    EncodeSingleString(input_string, 0, OffsetType::BYTE, &encoding);
    PostProcess(&encoding, nullptr, add_special_tokens, encodings);
  } else {
    Encoding pair_encoding;
    const auto& input_string_pair =
        boost::get<std::pair<InputString, InputString>>(encode_input);
    EncodeSingleString(input_string_pair.first, 0, OffsetType::BYTE, &encoding);
    EncodeSingleString(
        input_string_pair.second, 1, OffsetType::BYTE, &pair_encoding);
    PostProcess(&encoding, &pair_encoding, add_special_tokens, encodings);
  }
}

void Tokenizer::EncodeBatchStrings(
    const std::vector<EncodeInput>& batch_encode_input,
    bool add_special_tokens,
    std::vector<Encoding>* encodings) const {
  encodings->resize(batch_encode_input.size());
#ifdef WITH_OMP
#pragma omp parallel for
#endif
  for (int i = 0; i < batch_encode_input.size(); ++i) {
    EncodePairStrings(
        batch_encode_input[i], add_special_tokens, &(*encodings)[i]);
  }
  if (use_padding_) {
    PadEncodings(encodings, pad_method_);
  }
}

void Tokenizer::EncodePairStringsCharOffsets(const EncodeInput& encode_input,
                                             bool add_special_tokens,
                                             Encoding* encodings) const {
  const auto& input_string = boost::get<InputString>(&encode_input);
  const auto& input_string_pair =
      boost::get<std::pair<InputString, InputString>>(&encode_input);
  Encoding encoding;
  Encoding pair_encoding;
  if (input_string != nullptr) {
    EncodeSingleString(*input_string, 0, OffsetType::CHAR, &encoding);
  } else {
    EncodeSingleString(
        input_string_pair->first, 0, OffsetType::CHAR, &encoding);
    EncodeSingleString(
        input_string_pair->second, 1, OffsetType::CHAR, &pair_encoding);
  }
  PostProcess(&encoding, &pair_encoding, add_special_tokens, encodings);
}

void Tokenizer::EncodeBatchStringsCharOffsets(
    const std::vector<EncodeInput>& batch_encode_input,
    bool add_special_tokens,
    std::vector<Encoding>* encodings) const {
  encodings->resize(batch_encode_input.size());
#ifdef WITH_OMP
#pragma omp parallel for
#endif
  for (int i = 0; i < batch_encode_input.size(); ++i) {
    Encoding encoding;
    EncodePairStringsCharOffsets(
        batch_encode_input[i], add_special_tokens, &encoding);
    (*encodings)[i] = std::move(encoding);
  }
  if (use_padding_) {
    PadEncodings(encodings, pad_method_);
  }
}

void Tokenizer::EncodeSingleText(
    const std::vector<std::string>& pretokenized_texts,
    uint type_id,
    OffsetType offset_type,
    Encoding* encoding) const {
  std::vector<Encoding> encodings;
  for (uint i = 0; i < pretokenized_texts.size(); ++i) {
    encodings.emplace_back(
        EncodeTextToEncoding({i}, type_id, offset_type, pretokenized_texts[i]));
  }
  *encoding = Encoding::Merge(encodings, false);
}

void Tokenizer::EncodeSingleText(const std::string& raw_text,
                                 uint type_id,
                                 OffsetType offset_type,
                                 Encoding* encodings) const {
  *encodings = EncodeTextToEncoding({}, type_id, offset_type, raw_text);
}

Encoding Tokenizer::EncodeTextToEncoding(const std::vector<uint>& word_idx,
                                         uint type_id,
                                         OffsetType offset_type,
                                         const std::string& text) const {
  pretokenizers::PreTokenizedString pretokenized;
  added_vocabulary_.ExtractAndNormalize(normalizer_.get(), text, &pretokenized);
  DoPreTokenize(&pretokenized);
  Encoding encoding;
  DoTokenize(&pretokenized, type_id, word_idx, offset_type, &encoding);
  return encoding;
}

// Instantiate normalizers
template void Tokenizer::SetNormalizer(const normalizers::BertNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::LowercaseNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NFCNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NFKCNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NFDNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NFKDNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NmtNormalizer&);
// TODO(zhoushunjie): Need to implement PrecompiledNormalizer later
// template void Tokenizer::SetNormalizer(const
// normalizers::PrecompiledNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::ReplaceNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::SequenceNormalizer&);
template void Tokenizer::SetNormalizer(
    const normalizers::StripAccentsNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::StripNormalizer&);

// Instantiate pretokenizers
template void Tokenizer::SetPreTokenizer(
    const pretokenizers::BertPreTokenizer&);
template void Tokenizer::SetPreTokenizer(const pretokenizers::Whitespace&);

// Instantiate models
template Tokenizer::Tokenizer(const models::WordPiece&);
template void Tokenizer::SetModel(const models::WordPiece&);

// Instantiate processors
template void Tokenizer::SetPostProcessor(
    const postprocessors::BertPostProcessor&);

}  // core
}  // tokenizers

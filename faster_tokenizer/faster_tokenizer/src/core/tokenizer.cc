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

#include <fstream>
#include "glog/logging.h"

#include "core/added_vocabulary.h"
#include "core/base.h"
#include "core/encoding.h"
#include "core/tokenizer.h"

#include "decoders/decoders.h"
#include "models/models.h"
#include "normalizers/normalizers.h"
#include "postprocessors/postprocessors.h"
#include "pretokenizers/pretokenizers.h"

#ifdef WITH_OMP
#include <omp.h>
#endif

namespace paddlenlp {
namespace faster_tokenizer {
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
                                uint32_t pad_id,
                                uint32_t pad_type_id,
                                const std::string& pad_token,
                                uint32_t* length,
                                uint32_t* pad_to_multiple_of) {
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
    pad_method_.pad_to_multiple_of_ = *pad_to_multiple_of;
  } else {
    pad_method_.pad_to_multiple_of_ = 0;
  }
}
void Tokenizer::DisablePadMethod() { use_padding_ = false; }

PadMethod Tokenizer::GetPadMethod() const { return pad_method_; }

models::Model* Tokenizer::GetModelPtr() const { return model_.get(); }

void Tokenizer::ReleasePostProcessor() { post_processor_ = nullptr; }

postprocessors::PostProcessor* Tokenizer::GetPostProcessorPtr() const {
  return post_processor_.get();
}

void Tokenizer::ReleaseDecoder() { decoder_ = nullptr; }

decoders::Decoder* Tokenizer::GetDecoderPtr() const { return decoder_.get(); }

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

bool Tokenizer::TokenToId(const std::string& token, uint32_t* id) const {
  return added_vocabulary_.TokenToId(token, *model_, id);
}

bool Tokenizer::IdToToken(uint32_t id, std::string* token) const {
  return added_vocabulary_.IdToToken(id, *model_, token);
}

bool Tokenizer::DoTokenize(pretokenizers::PreTokenizedString* pretokenized,
                           uint32_t type_id,
                           const std::vector<uint32_t>& word_idx,
                           OffsetType offset_type,
                           Encoding* encoding) const {
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
                     uint32_t type_id,
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
  uint32_t type_id_;
  OffsetType offset_type_;
  Encoding* encodings_;
};

void Tokenizer::EncodeSingleString(const InputString& input_string,
                                   uint32_t type_id,
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
                                  Encoding* encodings,
                                  bool add_special_tokens) const {
  Encoding encoding;
  if (encode_input.type() == typeid(InputString)) {
    const auto& input_string = boost::get<InputString>(encode_input);
    EncodeSingleString(input_string, 0, OffsetType::CHAR, &encoding);
    PostProcess(&encoding, nullptr, add_special_tokens, encodings);
  } else {
    Encoding pair_encoding;
    const auto& input_string_pair =
        boost::get<std::pair<InputString, InputString>>(encode_input);
    EncodeSingleString(input_string_pair.first, 0, OffsetType::CHAR, &encoding);
    EncodeSingleString(
        input_string_pair.second, 1, OffsetType::CHAR, &pair_encoding);
    PostProcess(&encoding, &pair_encoding, add_special_tokens, encodings);
  }
}

void Tokenizer::EncodeBatchStrings(
    const std::vector<EncodeInput>& batch_encode_input,
    std::vector<Encoding>* encodings,
    bool add_special_tokens) const {
  encodings->resize(batch_encode_input.size());
#ifdef WITH_OMP
// (TODO:zhoushunjie): Simply use the batch size to estimate the workload of
// tokenization.
// Use workload to determine whether create omp threads. Need to optimize the
// workload estimation.
#pragma omp parallel for if (batch_encode_input.size() >= 4 &&               \
                                                     omp_get_num_threads() > \
                                                                         1)
#endif
  for (int i = 0; i < batch_encode_input.size(); ++i) {
    EncodePairStrings(
        batch_encode_input[i], &(*encodings)[i], add_special_tokens);
  }
  if (use_padding_) {
    PadEncodings(encodings, pad_method_);
  }
}

void Tokenizer::EncodePairStringsCharOffsets(const EncodeInput& encode_input,
                                             Encoding* encodings,
                                             bool add_special_tokens) const {
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
    std::vector<Encoding>* encodings,
    bool add_special_tokens) const {
  encodings->resize(batch_encode_input.size());
#ifdef WITH_OMP
// (TODO:zhoushunjie): Simply use the batch size to estimate the workload of
// tokenization.
// Use workload to determine whether create omp threads. Need to optimize the
// workload estimation.
#pragma omp parallel for if (batch_encode_input.size() >= 4 &&               \
                                                     omp_get_num_threads() > \
                                                                         1)
#endif
  for (int i = 0; i < batch_encode_input.size(); ++i) {
    Encoding encoding;
    EncodePairStringsCharOffsets(
        batch_encode_input[i], &encoding, add_special_tokens);
    (*encodings)[i] = std::move(encoding);
  }
  if (use_padding_) {
    PadEncodings(encodings, pad_method_);
  }
}

void Tokenizer::EncodeSingleText(
    const std::vector<std::string>& pretokenized_texts,
    uint32_t type_id,
    OffsetType offset_type,
    Encoding* encoding) const {
  std::vector<Encoding> encodings;
  for (uint32_t i = 0; i < pretokenized_texts.size(); ++i) {
    encodings.emplace_back(
        EncodeTextToEncoding({i}, type_id, offset_type, pretokenized_texts[i]));
  }
  *encoding = Encoding::Merge(encodings, false);
}

void Tokenizer::EncodeSingleText(const std::string& raw_text,
                                 uint32_t type_id,
                                 OffsetType offset_type,
                                 Encoding* encodings) const {
  *encodings = EncodeTextToEncoding({}, type_id, offset_type, raw_text);
}

Encoding Tokenizer::EncodeTextToEncoding(const std::vector<uint32_t>& word_idx,
                                         uint32_t type_id,
                                         OffsetType offset_type,
                                         const std::string& text) const {
  pretokenizers::PreTokenizedString pretokenized;
  added_vocabulary_.ExtractAndNormalize(normalizer_.get(), text, &pretokenized);
  DoPreTokenize(&pretokenized);
  Encoding encoding;
  DoTokenize(&pretokenized, type_id, word_idx, offset_type, &encoding);
  return encoding;
}
const AddedVocabulary& Tokenizer::GetAddedVocabulary() const {
  return added_vocabulary_;
}

void Tokenizer::Save(const std::string& path, bool pretty) const {
  std::string json_str;
  ToJsonStr(&json_str, pretty);
  std::ofstream fout(path);
  fout << json_str;
}

void Tokenizer::ToJsonStr(std::string* json_str, bool pretty) const {
  int indent = -1;
  if (pretty) {
    indent = 2;
  }
  nlohmann::json j = *this;
  *json_str = j.dump(indent);
}

Tokenizer Tokenizer::LoadFromFile(const std::string& json_path) {
  std::ifstream fin(json_path);
  nlohmann::json j;
  fin >> j;
  Tokenizer tokenizer;
  j.get_to(tokenizer);
  return tokenizer;
}

Tokenizer Tokenizer::LoadFromStr(const std::string& json_str) {
  auto jo = nlohmann::json::parse(json_str);
  Tokenizer tokenizer;
  jo.get_to(tokenizer);
  return tokenizer;
}

void Tokenizer::Decode(const std::vector<uint32_t>& token_ids,
                       std::string* result,
                       bool skip_special_tokens) const {
  // Get tokens
  std::vector<std::string> tokens;
  std::string token;
  for (int i = 0; i < token_ids.size(); ++i) {
    IdToToken(token_ids[i], &token);
    if (!added_vocabulary_.IsSpecialToken(token) || !skip_special_tokens) {
      tokens.push_back(token);
    }
  }
  if (decoder_ != nullptr) {
    (*decoder_)(tokens, result);
  } else {
    for (int i = 0; i < tokens.size(); ++i) {
      if (i > 0) {
        *result += " ";
      }
      *result += tokens[i];
    }
  }
}

void Tokenizer::DecodeBatch(
    const std::vector<std::vector<uint32_t>>& batch_token_ids,
    std::vector<std::string>* results,
    bool skip_special_tokens) const {
  results->resize(batch_token_ids.size());
#ifdef WITH_OMP
// (TODO:zhoushunjie): Simply use the batch size to estimate the workload of
// tokenization.
// Use workload to determine whether create omp threads. Need to optimize the
// workload estimation.
#pragma omp parallel for if (batch_token_ids.size() >= 4 && \
                                                  omp_get_num_threads() > 1)
#endif
  for (int i = 0; i < batch_token_ids.size(); ++i) {
    Decode(batch_token_ids[i], &(*results)[i], skip_special_tokens);
  }
}

bool Tokenizer::GetUseTruncation() const { return use_truncation_; }

bool Tokenizer::GetUsePadding() const { return use_padding_; }

void to_json(nlohmann::json& j, const Tokenizer& tokenizer) {
  j = {
      {"added_tokens", tokenizer.added_vocabulary_},
  };

  j["truncation"] = nullptr;
  if (tokenizer.use_truncation_) {
    j["truncation"] = tokenizer.trunc_method_;
  }

  j["padding"] = nullptr;
  if (tokenizer.use_padding_) {
    j["padding"] = tokenizer.pad_method_;
  }

  j["normalizer"] = nullptr;
  if (tokenizer.normalizer_ != nullptr) {
    if (typeid(*tokenizer.normalizer_.get()) ==
        typeid(normalizers::BertNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::BertNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::ReplaceNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::ReplaceNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::StripNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::StripNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::StripAccentsNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::StripAccentsNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::NFCNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::NFCNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::NFDNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::NFDNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::NFKCNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::NFKCNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::NFKDNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::NFKDNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::NmtNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::NmtNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::LowercaseNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::LowercaseNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::SequenceNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::SequenceNormalizer*>(
          tokenizer.normalizer_.get());
    } else if (typeid(*tokenizer.normalizer_.get()) ==
               typeid(normalizers::PrecompiledNormalizer)) {
      j["normalizer"] = *dynamic_cast<normalizers::PrecompiledNormalizer*>(
          tokenizer.normalizer_.get());
    }
  }

  j["pretokenizer"] = nullptr;
  if (tokenizer.pretokenizer_ != nullptr) {
    if (typeid(*tokenizer.pretokenizer_.get()) ==
        typeid(pretokenizers::BertPreTokenizer)) {
      j["pretokenizer"] = *dynamic_cast<pretokenizers::BertPreTokenizer*>(
          tokenizer.pretokenizer_.get());
    } else if (typeid(*tokenizer.pretokenizer_.get()) ==
               typeid(pretokenizers::MetaSpacePreTokenizer)) {
      j["pretokenizer"] = *dynamic_cast<pretokenizers::MetaSpacePreTokenizer*>(
          tokenizer.pretokenizer_.get());
    } else if (typeid(*tokenizer.pretokenizer_.get()) ==
               typeid(pretokenizers::WhitespacePreTokenizer)) {
      j["pretokenizer"] = *dynamic_cast<pretokenizers::WhitespacePreTokenizer*>(
          tokenizer.pretokenizer_.get());
    } else if (typeid(*tokenizer.pretokenizer_.get()) ==
               typeid(pretokenizers::SequencePreTokenizer)) {
      j["pretokenizer"] = *dynamic_cast<pretokenizers::SequencePreTokenizer*>(
          tokenizer.pretokenizer_.get());
    }
  }

  j["model"] = nullptr;
  if (tokenizer.model_ != nullptr) {
    if (typeid(*tokenizer.model_.get()) == typeid(models::WordPiece)) {
      j["model"] = *dynamic_cast<models::WordPiece*>(tokenizer.model_.get());
    } else if (typeid(*tokenizer.model_.get()) ==
               typeid(models::FasterWordPiece)) {
      j["model"] =
          *dynamic_cast<models::FasterWordPiece*>(tokenizer.model_.get());
    } else if (typeid(*tokenizer.model_.get()) == typeid(models::BPE)) {
      j["model"] = *dynamic_cast<models::BPE*>(tokenizer.model_.get());
    } else if (typeid(*tokenizer.model_.get()) == typeid(models::Unigram)) {
      j["model"] = *dynamic_cast<models::Unigram*>(tokenizer.model_.get());
    }
  }

  j["postprocessor"] = nullptr;
  if (tokenizer.post_processor_ != nullptr) {
    if (typeid(*tokenizer.post_processor_.get()) ==
        typeid(postprocessors::BertPostProcessor)) {
      j["postprocessor"] = *dynamic_cast<postprocessors::BertPostProcessor*>(
          tokenizer.post_processor_.get());
    } else if (typeid(*tokenizer.post_processor_.get()) ==
               typeid(postprocessors::TemplatePostProcessor)) {
      j["postprocessor"] =
          *dynamic_cast<postprocessors::TemplatePostProcessor*>(
              tokenizer.post_processor_.get());
    }
  }

  j["decoder"] = nullptr;
  if (tokenizer.decoder_ != nullptr) {
    if (typeid(*tokenizer.decoder_.get()) == typeid(decoders::WordPiece)) {
      j["decoder"] =
          *dynamic_cast<decoders::WordPiece*>(tokenizer.decoder_.get());
    }
  }
}

void from_json(const nlohmann::json& j, Tokenizer& tokenizer) {
  // deserialize normalizer_
  try {
    const auto& normalizer = j.at("normalizer");
    if (!normalizer.is_null()) {
      if (normalizer.at("type") == "BertNormalizer") {
        normalizers::BertNormalizer bert_normalizer;
        normalizer.get_to(bert_normalizer);
        tokenizer.SetNormalizer(bert_normalizer);
      } else if (normalizer.at("type") == "ReplaceNormalizer") {
        normalizers::ReplaceNormalizer replace_normalizer;
        normalizer.get_to(replace_normalizer);
        tokenizer.SetNormalizer(replace_normalizer);
      } else if (normalizer.at("type") == "StripNormalizer") {
        normalizers::StripNormalizer strip_normalizer;
        normalizer.get_to(strip_normalizer);
        tokenizer.SetNormalizer(strip_normalizer);
      } else if (normalizer.at("type") == "StripAccentsNormalizer") {
        normalizers::StripAccentsNormalizer strip_normalizer;
        normalizer.get_to(strip_normalizer);
        tokenizer.SetNormalizer(strip_normalizer);
      } else if (normalizer.at("type") == "NFCNormalizer") {
        normalizers::NFCNormalizer unicode_normalizer;
        normalizer.get_to(unicode_normalizer);
        tokenizer.SetNormalizer(unicode_normalizer);
      } else if (normalizer.at("type") == "NFDNormalizer") {
        normalizers::NFDNormalizer unicode_normalizer;
        normalizer.get_to(unicode_normalizer);
        tokenizer.SetNormalizer(unicode_normalizer);
      } else if (normalizer.at("type") == "NFKCNormalizer") {
        normalizers::NFKCNormalizer unicode_normalizer;
        normalizer.get_to(unicode_normalizer);
        tokenizer.SetNormalizer(unicode_normalizer);
      } else if (normalizer.at("type") == "NFKDNormalizer") {
        normalizers::NFKDNormalizer unicode_normalizer;
        normalizer.get_to(unicode_normalizer);
        tokenizer.SetNormalizer(unicode_normalizer);
      } else if (normalizer.at("type") == "NmtNormalizer") {
        normalizers::NmtNormalizer unicode_normalizer;
        normalizer.get_to(unicode_normalizer);
        tokenizer.SetNormalizer(unicode_normalizer);
      } else if (normalizer.at("type") == "LowercaseNormalizer") {
        normalizers::LowercaseNormalizer unicode_normalizer;
        normalizer.get_to(unicode_normalizer);
        tokenizer.SetNormalizer(unicode_normalizer);
      } else if (normalizer.at("type") == "SequenceNormalizer") {
        normalizers::SequenceNormalizer unicode_normalizer;
        normalizer.get_to(unicode_normalizer);
        tokenizer.SetNormalizer(unicode_normalizer);
      } else if (normalizer.at("type") == "PrecompiledNormalizer") {
        normalizers::PrecompiledNormalizer precompiled_normalizer;
        normalizer.get_to(precompiled_normalizer);
        tokenizer.SetNormalizer(precompiled_normalizer);
      }
    }

    // deserialize pretokenizer_
    const auto& pretokenizer = j.at("pretokenizer");
    if (!pretokenizer.is_null()) {
      if (pretokenizer.at("type") == "BertPreTokenizer") {
        pretokenizers::BertPreTokenizer bert_pretokenizer;
        tokenizer.SetPreTokenizer(bert_pretokenizer);
      } else if (pretokenizer.at("type") == "MetaSpacePreTokenizer") {
        pretokenizers::MetaSpacePreTokenizer meta_pretokenizer;
        pretokenizer.get_to(meta_pretokenizer);
        tokenizer.SetPreTokenizer(meta_pretokenizer);
      } else if (pretokenizer.at("type") == "WhitespacePreTokenizer") {
        pretokenizers::WhitespacePreTokenizer whitespace_pretokenizer;
        tokenizer.SetPreTokenizer(whitespace_pretokenizer);
      } else if (pretokenizer.at("type") == "SequencePreTokenizer") {
        pretokenizers::SequencePreTokenizer sequence_pretokenizer;
        pretokenizer.get_to(sequence_pretokenizer);
        tokenizer.SetPreTokenizer(sequence_pretokenizer);
      }
    }

    // deserialize model_
    const auto& model = j.at("model");
    if (!model.is_null()) {
      if (model.at("type") == "WordPiece") {
        models::WordPiece wordpiece;
        model.get_to(wordpiece);
        tokenizer.SetModel(wordpiece);
      } else if (model.at("type") == "FasterWordPiece") {
        models::FasterWordPiece wordpiece;
        model.get_to(wordpiece);
        tokenizer.SetModel(wordpiece);
      } else if (model.at("type") == "BPE") {
        models::BPE bpe;
        model.get_to(bpe);
        tokenizer.SetModel(bpe);
      } else if (model.at("type") == "Unigram") {
        models::Unigram unigram;
        model.get_to(unigram);
        tokenizer.SetModel(unigram);
      }
    }

    // deserialize post_processor_
    const auto& post_processor = j.at("postprocessor");
    if (!post_processor.is_null()) {
      if (post_processor.at("type") == "BertPostProcessor") {
        postprocessors::BertPostProcessor bert_postprocessor;
        post_processor.get_to(bert_postprocessor);
        tokenizer.SetPostProcessor(bert_postprocessor);
      } else if (post_processor.at("type") == "TemplateProcessing") {
        postprocessors::TemplatePostProcessor template_postprocessor;
        post_processor.get_to(template_postprocessor);
        tokenizer.SetPostProcessor(template_postprocessor);
      }
    }

    // deserialize trunc_method_
    const auto& trunc_method = j.at("truncation");
    if (!trunc_method.is_null()) {
      tokenizer.use_truncation_ = true;
      trunc_method.get_to(tokenizer.trunc_method_);
    } else {
      tokenizer.use_truncation_ = false;
    }

    // deserialize pad_method_
    const auto& pad_method = j.at("padding");
    if (!pad_method.is_null()) {
      tokenizer.use_padding_ = true;
      pad_method.get_to(tokenizer.pad_method_);
    } else {
      tokenizer.use_padding_ = false;
    }

    // deserialize added_vocabulary_
    const auto& added_tokens = j.at("added_tokens");
    core::AddedTokenWithId added_token_with_id;
    std::vector<AddedToken> tokens(added_tokens.size());
    for (int i = 0; i < added_tokens.size(); ++i) {
      added_tokens[i].get_to(added_token_with_id);
      tokens[i] = added_token_with_id.added_token_;
    }
    tokenizer.AddSpecialTokens(tokens);

    const auto& decoder = j.at("decoder");
    if (!decoder.is_null()) {
      if (decoder.at("type") == "WordPiece") {
        decoders::WordPiece wordpiece_decoder;
        decoder.get_to(wordpiece_decoder);
        tokenizer.SetDecoder(wordpiece_decoder);
      }
    }

  } catch (nlohmann::json::out_of_range& e) {
    VLOG(0) << e.what();
  }
}
// Instantiate normalizers
template void Tokenizer::SetNormalizer(const normalizers::BertNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::LowercaseNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NFCNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NFKCNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NFDNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NFKDNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::NmtNormalizer&);
template void Tokenizer::SetNormalizer(
    const normalizers::PrecompiledNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::ReplaceNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::SequenceNormalizer&);
template void Tokenizer::SetNormalizer(
    const normalizers::StripAccentsNormalizer&);
template void Tokenizer::SetNormalizer(const normalizers::StripNormalizer&);

// Instantiate pretokenizers
template void Tokenizer::SetPreTokenizer(
    const pretokenizers::BertPreTokenizer&);
template void Tokenizer::SetPreTokenizer(
    const pretokenizers::WhitespacePreTokenizer&);
template void Tokenizer::SetPreTokenizer(
    const pretokenizers::MetaSpacePreTokenizer&);
template void Tokenizer::SetPreTokenizer(
    const pretokenizers::SequencePreTokenizer&);

// Instantiate models
template Tokenizer::Tokenizer(const models::WordPiece&);
template void Tokenizer::SetModel(const models::WordPiece&);
template Tokenizer::Tokenizer(const models::FasterWordPiece&);
template void Tokenizer::SetModel(const models::FasterWordPiece&);
template Tokenizer::Tokenizer(const models::BPE&);
template void Tokenizer::SetModel(const models::BPE&);
template Tokenizer::Tokenizer(const models::Unigram&);
template void Tokenizer::SetModel(const models::Unigram&);

// Instantiate processors
template void Tokenizer::SetPostProcessor(
    const postprocessors::BertPostProcessor&);
template void Tokenizer::SetPostProcessor(
    const postprocessors::TemplatePostProcessor&);

// Instantiate Decoder
template void Tokenizer::SetDecoder(const decoders::WordPiece& decoder);
}  // namespace core
}  // namespace faster_tokenizer
}  // namespace paddlenlp

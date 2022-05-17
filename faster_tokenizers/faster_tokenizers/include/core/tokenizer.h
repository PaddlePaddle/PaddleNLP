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
#include <memory>  // For shared_ptr
#include <vector>

#include "boost/variant.hpp"
#include "core/added_vocabulary.h"
#include "core/base.h"
#include "nlohmann/json.hpp"

namespace tokenizers {

namespace normalizers {

class Normalizer;
class NormalizedString;

}  // normalizers

namespace pretokenizers {

class PreTokenizer;
class PreTokenizedString;

}  // pretokenizers

namespace models {
class Model;
}  // models

namespace postprocessors {
class PostProcessor;
}  // postprocessors

namespace decoders {
class Decoder;
};

namespace core {

class AddedVocabulary;
class Encoding;

using InputString = boost::variant<std::string, std::vector<std::string>>;
using EncodeInput =
    boost::variant<InputString, std::pair<InputString, InputString>>;

class Tokenizer {
public:
  Tokenizer()
      : model_(nullptr),
        normalizer_(nullptr),
        pretokenizer_(nullptr),
        post_processor_(nullptr),
        decoder_(nullptr),
        use_padding_(true),
        use_truncation_(true) {}
  template <typename ModelType>
  Tokenizer(const ModelType& model)
      : model_(std::make_shared<ModelType>(model)),
        normalizer_(nullptr),
        pretokenizer_(nullptr),
        post_processor_(nullptr),
        decoder_(nullptr),
        use_padding_(true),
        use_truncation_(true) {}

  template <typename NormalizerType>
  void SetNormalizer(const NormalizerType& normalizer) {
    normalizer_ = std::make_shared<NormalizerType>(normalizer);
  }
  void ReleaseNormaizer();
  normalizers::Normalizer* GetNormalizerPtr() const;

  template <typename PreTokenizerType>
  void SetPreTokenizer(const PreTokenizerType& pretokenizer) {
    pretokenizer_ = std::make_shared<PreTokenizerType>(pretokenizer);
  }
  void ReleasePreTokenizer();
  pretokenizers::PreTokenizer* GetPreTokenizer() const;

  template <typename ModelType>
  void SetModel(const ModelType& model) {
    model_ = std::make_shared<ModelType>(model);
  }
  models::Model* GetModelPtr() const;

  template <typename PostProcessorType>
  void SetPostProcessor(const PostProcessorType& post_processor) {
    post_processor_ = std::make_shared<PostProcessorType>(post_processor);
  }
  void ReleasePostProcessor();
  postprocessors::PostProcessor* GetPostProcessorPtr() const;

  template <typename DecoderType>
  void SetDecoder(const DecoderType& decoder) {
    decoder_ = std::make_shared<DecoderType>(decoder);
  }
  void ReleaseDecoder();
  decoders::Decoder* GetDecoderPtr() const;

  void SetTruncMethod(const TruncMethod& trunc_method);
  void DisableTruncMethod();
  void EnableTruncMethod(size_t max_len,
                         size_t stride,
                         Direction direction,
                         TruncStrategy strategy);
  TruncMethod GetTruncMethod() const;

  void SetPadMethod(const PadMethod& pad_method);
  void DisablePadMethod();
  void EnablePadMethod(Direction direction,
                       uint pad_id,
                       uint pad_type_id,
                       const std::string& pad_token,
                       uint* length,
                       uint* pad_to_multiple_of);
  PadMethod GetPadMethod() const;

  Vocab GetVocab(bool with_added_vocabulary = true) const;
  size_t GetVocabSize(bool with_added_vocabulary = true) const;
  bool TokenToId(const std::string& token, uint* id) const;
  bool IdToToken(uint id, std::string* token) const;
  size_t AddTokens(const std::vector<AddedToken>& tokens);
  size_t AddSpecialTokens(const std::vector<AddedToken>& tokens);
  bool DoTokenize(pretokenizers::PreTokenizedString* pretokenized,
                  uint type_id,
                  const std::vector<uint>& word_idx,
                  OffsetType offset_type,
                  Encoding* encoding) const;
  bool DoPreTokenize(pretokenizers::PreTokenizedString* pretokenized) const;

  void EncodeSingleString(const InputString& input_string,
                          uint type_id,
                          OffsetType offset_type,
                          Encoding* encodings) const;
  void EncodePairStrings(const EncodeInput& encode_input,
                         Encoding* encodings,
                         bool add_special_tokens = true) const;
  void EncodePairStringsCharOffsets(const EncodeInput& encode_input,
                                    Encoding* encodings,
                                    bool add_special_tokens = true) const;
  void PostProcess(Encoding* encoding,
                   Encoding* pair_encoding,
                   bool add_special_tokens,
                   Encoding* result_encoding) const;

  void EncodeBatchStrings(const std::vector<EncodeInput>& batch_encode_input,
                          std::vector<Encoding>* encodings,
                          bool add_special_tokens = true) const;

  void EncodeBatchStringsCharOffsets(
      const std::vector<EncodeInput>& batch_encode_input,
      std::vector<Encoding>* encodings,
      bool add_special_tokens = true) const;

  // Encode single text which is already pretokenized.
  void EncodeSingleText(const std::vector<std::string>& pretokenized_texts,
                        uint type_id,
                        OffsetType offset_type,
                        Encoding* encodings) const;
  // Encode single raw text
  void EncodeSingleText(const std::string& raw_text,
                        uint type_id,
                        OffsetType offset_type,
                        Encoding* encodings) const;
  const AddedVocabulary& GetAddedVocabulary() const;
  void Save(const std::string& json_path, bool pretty = true) const;
  void ToJsonStr(std::string* json_str, bool pretty = true) const;

  // Create a tokenzier from json path
  static Tokenizer LoadFromFile(const std::string& json_path);
  static Tokenizer LoadFromStr(const std::string& json_str);

  bool GetUseTruncation() const;
  bool GetUsePadding() const;

  // Decode: From tokens to a complete string
  void Decode(const std::vector<uint>& token_ids,
              std::string* result,
              bool skip_special_tokens = true) const;
  void DecodeBatch(const std::vector<std::vector<uint>>& batch_token_ids,
                   std::vector<std::string>* results,
                   bool skip_special_tokens = true) const;

private:
  Encoding EncodeTextToEncoding(const std::vector<uint>& word_idx,
                                uint type_id,
                                OffsetType offset_type,
                                const std::string& text) const;
  // All member of Tokenizer
  std::shared_ptr<normalizers::Normalizer> normalizer_;
  std::shared_ptr<pretokenizers::PreTokenizer> pretokenizer_;
  std::shared_ptr<models::Model> model_;
  std::shared_ptr<postprocessors::PostProcessor> post_processor_;
  std::shared_ptr<decoders::Decoder> decoder_;

  TruncMethod trunc_method_;
  PadMethod pad_method_;
  AddedVocabulary added_vocabulary_;
  bool use_truncation_;
  bool use_padding_;
  // TODO(zhoushunjie): Implement Decoder later.
  friend void to_json(nlohmann::json& j, const Tokenizer& tokenizer);
  friend void from_json(const nlohmann::json& j, Tokenizer& tokenizer);
};

}  // core
}  // tokenizers

// Copyright 2016 Google Inc.
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>
#include "utils/string_view.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

// Copy from https://github.com/google/sentencepiece/blob/master/src/freelist.h
// Simple FreeList that allocates a chunk of T at once.
template <class T>
class FreeList {
public:
  FreeList() = delete;
  explicit FreeList(size_t chunk_size) : chunk_size_(chunk_size) {}
  virtual ~FreeList() {
    for (auto &chunk : freelist_) delete[] chunk;
  }

  // `Free` doesn't free the object but reuse the allocated memory chunks.
  void Free() {
    const int size = std::min<int>(chunk_index_ + 1, freelist_.size());
    for (int i = 0; i < size; ++i) {
      T *chunk = freelist_[i];
      memset(static_cast<void *>(chunk), 0, sizeof(*chunk) * chunk_size_);
    }
    chunk_index_ = 0;
    element_index_ = 0;
  }

  // Returns the number of allocated elements.
  size_t size() const { return chunk_size_ * chunk_index_ + element_index_; }

  void swap(FreeList<T> &other) {
    std::swap(freelist_, other.freelist_);
    std::swap(element_index_, other.element_index_);
    std::swap(chunk_index_, other.chunk_index_);
    std::swap(chunk_size_, other.chunk_size_);
  }

  // Returns the element as an array.
  T *operator[](size_t index) const {
    return freelist_[index / chunk_size_] + index % chunk_size_;
  }

  // Allocates new element.
  T *Allocate() {
    if (element_index_ >= chunk_size_) {
      ++chunk_index_;
      element_index_ = 0;
    }

    if (chunk_index_ == freelist_.size()) {
      T *chunk = new T[chunk_size_];
      memset(static_cast<void *>(chunk), 0, sizeof(*chunk) * chunk_size_);
      freelist_.push_back(chunk);
    }

    T *result = freelist_[chunk_index_] + element_index_;
    ++element_index_;

    return result;
  }

private:
  std::vector<T *> freelist_;

  // The last element is stored at freelist_[chunk_index_][element_index_]
  size_t element_index_ = 0;
  size_t chunk_index_ = 0;
  size_t chunk_size_ = 0;  // Do not modify except in swap()
};


// Copy from
// https://github.com/google/sentencepiece/blob/master/src/unigram_model.h
class Lattice {
public:
  Lattice();
  virtual ~Lattice();

  struct Node {
    utils::simple_string_view piece;  // Sentence piece representation.
    uint32_t pos;                     // Unicode position in the sentence.
    uint32_t length;                  // Unicode length, not UT8 byte.
    uint32_t node_id;                 // unique id in the current lattice.
    int id;                           // vocab id. (maybe -1 for UNK)
    float score;                      // logprob of this sentencepiece.
    float backtrace_score;            // backtrace info used in Viterbi.
    Node *prev;                       // best previous node on Viterbi path.

    std::string DebugString() const;
  };

  // Returns bos node.
  Node *bos_node() const;

  // Returns eos node.
  Node *eos_node() const;

  // Returns nodes starting at |pos|.
  const std::vector<Node *> &begin_nodes(int pos) const;

  // Returns nodes ending at |pos|.
  const std::vector<Node *> &end_nodes(int pos) const;

  // Returns Unicode character length.
  int size() const;

  // Returns multi-byte (utf8) length.
  int utf8_size() const;

  // Returns the substring of sentence. sentence[pos:]
  const char *surface(int pos) const;

  // Returns immutable sentence. The same as surface(0)
  const char *sentence() const;

  // Clears the lattice.
  void Clear();

  // Sets new sentence.
  void SetSentence(utils::simple_string_view sentence);

  // Inserts a new node at [pos, pos + length - 1].
  // After calling this method, The caller must set Node::score and Node::id.
  Node *Insert(int pos, int length);

  using LatticePathWithScore = std::pair<std::vector<Node *>, float>;

  // Returns Viterbi path. All nodes must be populated in advance.
  LatticePathWithScore Viterbi();

  // Runs forwards/backwards algorithm, returns vector with normalised
  // transition probs.
  std::vector<float> ForwardAlgorithm(float theta) const;
  std::vector<float> BackwardAlgorithm(float theta) const;

  // Returns n-best results.
  std::vector<LatticePathWithScore> NBest(size_t nbest_size,
                                          bool sample,
                                          float theta);

  // Samples one path from the lattice according to the
  // generation probability (Product of piece probabilities).
  // `theta` is a smoothing parameter.
  std::vector<Node *> Sample(float theta);

  // Calculates the entropy of the lattice.
  float CalculateEntropy(float theta) const;

  // Populates marginal probability of every node in this lattice.
  // |freq| is the frequency of the sentence.
  //  for (auto *node : all_nodes_) {
  //    (*expected)[node->id] += marginal_prob_of_node * freq;
  //  }
  // Returns the log-likelihood of this sentence.
  float PopulateMarginal(float freq, std::vector<float> *expected) const;

private:
  // Returns new node.
  // Lattice class has the ownership of the returned value.
  Node *NewNode();

  utils::simple_string_view sentence_;
  std::vector<const char *> surface_;
  std::vector<std::vector<Node *>> begin_nodes_;
  std::vector<std::vector<Node *>> end_nodes_;
  FreeList<Node> node_allocator_;
};


}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp

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

#include "utils/lattice.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <map>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "glog/logging.h"

#include "utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

// Size of nodes pre-allocated in Lattice.
constexpr size_t kPreallocateLatticeNodeSize = 1024;

constexpr float kEpsilon = 1e-7;

constexpr unsigned int kDefaultSeed = static_cast<unsigned int>(-1);
static std::atomic<unsigned int> g_seed(kDefaultSeed);

inline float LogSumExp(float x, float y, bool init_mode) {
  if (init_mode) {
    return y;
  }
  const float vmin = std::min(x, y);
  const float vmax = std::max(x, y);
  constexpr float kMinusLogEpsilon = 50;
  if (vmax > vmin + kMinusLogEpsilon) {
    return vmax;
  } else {
    return vmax + log(std::exp(static_cast<double>(vmin - vmax)) + 1.0);
  }
}

uint32_t GetRandomGeneratorSeed() {
  return g_seed == kDefaultSeed ? std::random_device{}() : g_seed.load();
}

std::mt19937 *GetRandomGenerator() {
  thread_local static std::mt19937 mt(GetRandomGeneratorSeed());
  return &mt;
}

inline float Gumbel() {
  const float kEpsilon = 1e-7;
  auto *mt = GetRandomGenerator();
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  float noise = -std::log(-(std::log(dis(*mt) + kEpsilon)));
  return noise;
}

Lattice::Lattice() : node_allocator_(kPreallocateLatticeNodeSize) {}
Lattice::~Lattice() {}

const std::vector<Lattice::Node *> &Lattice::begin_nodes(int pos) const {
  return begin_nodes_[pos];
}

const std::vector<Lattice::Node *> &Lattice::end_nodes(int pos) const {
  return end_nodes_[pos];
}

int Lattice::size() const {
  // -1 because surface_ may include the EOS.
  return std::max<int>(0, surface_.size() - 1);
}

int Lattice::utf8_size() const { return sentence_.size(); }

const char *Lattice::sentence() const { return sentence_.data(); }

const char *Lattice::surface(int pos) const { return surface_[pos]; }

Lattice::Node *Lattice::bos_node() const { return end_nodes_[0][0]; }

Lattice::Node *Lattice::eos_node() const { return begin_nodes_[size()][0]; }

Lattice::Node *Lattice::NewNode() {
  Node *node = node_allocator_.Allocate();
  node->node_id = node_allocator_.size() - 1;
  return node;
}

void Lattice::Clear() {
  begin_nodes_.clear();
  end_nodes_.clear();
  sentence_ = utils::simple_string_view("");
  surface_.clear();
  node_allocator_.Free();
}

void Lattice::SetSentence(utils::simple_string_view sentence) {
  Clear();

  sentence_ = sentence;
  surface_.reserve(sentence.size() + 1);

  while (!sentence.empty()) {
    const int mblen =
        std::min<int>(utils::OneCharLen(sentence.data()), sentence.size());
    surface_.push_back(sentence.data());
    sentence.remove_prefix(mblen);
  }
  surface_.push_back(sentence.data());

  const int len = size();
  begin_nodes_.resize(len + 1);
  end_nodes_.resize(len + 1);

  constexpr size_t kReservedNodeSize = 16;
  for (int i = 0; i <= len; ++i) {
    begin_nodes_[i].reserve(kReservedNodeSize);
    end_nodes_[i].reserve(kReservedNodeSize);
  }

  Node *bos = NewNode();
  bos->id = -1;
  bos->pos = 0;
  end_nodes_[0].push_back(bos);

  Node *eos = NewNode();
  eos->id = -1;
  eos->pos = len;
  begin_nodes_[len].push_back(eos);
}

Lattice::Node *Lattice::Insert(int pos, int length) {
  Node *node = NewNode();
  node->pos = pos;
  node->length = length;
  const int utf8_length =
      static_cast<int>(surface(pos + length) - surface(pos));
  node->piece = simple_string_view(surface(pos), utf8_length);
  begin_nodes_[pos].push_back(node);
  end_nodes_[pos + node->length].push_back(node);
  return node;
}

Lattice::LatticePathWithScore Lattice::Viterbi() {
  const int len = size();

  for (int pos = 0; pos <= len; ++pos) {
    for (Node *rnode : begin_nodes_[pos]) {
      rnode->prev = nullptr;
      float best_score = 0.0;
      Node *best_node = nullptr;
      for (Node *lnode : end_nodes_[pos]) {
        const float score = lnode->backtrace_score + rnode->score;
        if (best_node == nullptr || score > best_score) {
          best_node = lnode;
          best_score = score;
        }
      }
      if (best_node == nullptr) {
        LOG(ERROR) << "Failed to find the best path in Viterbi.";
        return {};
      }
      rnode->prev = best_node;
      rnode->backtrace_score = best_score;
    }
  }

  // backtrace
  std::vector<Node *> results;
  float score = begin_nodes(len)[0]->backtrace_score;
  for (Node *node = begin_nodes_[len][0]->prev; node->prev != nullptr;
       node = node->prev) {
    results.push_back(node);
  }

  std::reverse(results.begin(), results.end());

  LatticePathWithScore retval = {results, score};

  return retval;
}

std::vector<float> Lattice::ForwardAlgorithm(float inv_theta) const {
  const int len = size();
  std::vector<float> alpha(node_allocator_.size(), 0.0);

  for (int pos = 0; pos <= len; ++pos) {
    for (Node *rnode : begin_nodes_[pos]) {
      for (Node *lnode : end_nodes_[pos]) {
        alpha[rnode->node_id] =
            LogSumExp(alpha[rnode->node_id],
                      inv_theta * lnode->score + alpha[lnode->node_id],
                      lnode == end_nodes_[pos][0]);
      }
    }
  }

  return alpha;
}

std::vector<float> Lattice::BackwardAlgorithm(float inv_theta) const {
  const int len = size();
  std::vector<float> beta(node_allocator_.size(), 0.0);

  for (int pos = len; pos >= 0; --pos) {
    for (Node *lnode : end_nodes_[pos]) {
      for (Node *rnode : begin_nodes_[pos]) {
        beta[lnode->node_id] = LogSumExp(beta[lnode->node_id],
                                         rnode->score + beta[rnode->node_id],
                                         rnode == begin_nodes_[pos][0]);
      }
    }
  }

  return beta;
}

float Lattice::PopulateMarginal(float freq,
                                std::vector<float> *expected) const {
  if (expected == nullptr) return 0.0;

  const int len = size();

  // alpha and beta (accumulative log prob) in Forward Backward.
  // the index of alpha/beta is Node::node_id.

  const auto alpha = ForwardAlgorithm(1.0);
  const auto beta = BackwardAlgorithm(1.0);

  const float Z = alpha[begin_nodes_[len][0]->node_id];
  for (int pos = 0; pos < len; ++pos) {
    for (Node *node : begin_nodes_[pos]) {
      if (node->id >= 0) {
        // the index of |expected| is a Node::id, which is a vocabulary id.
        (*expected)[node->id] +=
            freq *
            std::exp(static_cast<double>(alpha[node->node_id] + node->score +
                                         beta[node->node_id] - Z));
      }
    }
  }

  return freq * Z;
}

float Lattice::CalculateEntropy(float inv_theta) const {
  const int len = size();

  // alpha[node_id] is the marginal prob of sequence up to start of node
  // H is entropy of sequence
  // the index of alpha/H is Node::node_id.
  std::vector<float> H(node_allocator_.size(), 0.0);

  // Populate the forward marginals to get the normalising constant
  const auto alpha = ForwardAlgorithm(inv_theta);

  // Now populate the forward entropies
  for (int pos = 0; pos <= len; ++pos) {
    for (Node *rnode : begin_nodes_[pos]) {
      for (Node *lnode : end_nodes_[pos]) {
        // Contribution each lnode makes = p(lnode) * (H(lnode) + log p(lnode))

        // We have to normalise p(lnode) by the marginal contribution it makes
        const float lnode_transition_prob =
            ((inv_theta * lnode->score) + alpha[lnode->node_id] -
             alpha[rnode->node_id]);
        H[rnode->node_id] += std::exp(lnode_transition_prob) *
                             (H[lnode->node_id] + lnode_transition_prob);
      }
    }
  }

  return -H[begin_nodes_[len][0]->node_id];
}

// The node structure to support A* algorithm in Lattice::NBest()
struct Hypothesis {
  Lattice::Node *node;
  Hypothesis *next;
  float fx;  // the priority to pop a new hypothesis from the priority queue.
  float gx;  // the sum of scores from EOS to the left-most node in x.
};

// Helper function for cloning a Hypothesis and the ones on their next paths.
// The graph structure is preserved.
//
//   to_clone:  the Hypothesis to clone.
//   clone_map: mapping between the old pointers and the new pointers.
//   allocator: allocate and own the cloned Hypothesis.
//
// Returns the cloned Hypothesis*. All Hypothesis on its "next" chain are also
// guaranteed to have been allocated via "allocator", and "clone_map" is updated
// with all new mappings.
Hypothesis *CloneHypAndDependents(
    const Hypothesis *to_clone,
    std::unordered_map<const Hypothesis *, Hypothesis *> *clone_map,
    FreeList<Hypothesis> *allocator) {
  Hypothesis *cloned = nullptr;
  Hypothesis **result_callback = &cloned;

  // Iteratively clone "to_clone" and its dependencies.
  // The new pointer will be written back to *result_callback.
  while (to_clone != nullptr) {
    // If "to_clone" has already been cloned before, we just look up the result.
    auto iter = clone_map->find(to_clone);
    if (iter != clone_map->end()) {
      *result_callback = iter->second;
      break;
    }

    // Allocate a new Hypothesis and copy the values.
    Hypothesis *new_hyp = allocator->Allocate();
    *new_hyp = *to_clone;
    *result_callback = new_hyp;
    clone_map->insert({to_clone, new_hyp});

    // Move on to clone "to_clone->next".
    to_clone = to_clone->next;
    result_callback = &(new_hyp->next);
    LOG(ERROR) << "Failed to find the best path in Viterbi.";
  }
  return cloned;
}

std::vector<Lattice::LatticePathWithScore> Lattice::NBest(size_t nbest_size,
                                                          bool sample,
                                                          float inv_theta) {
  if (nbest_size < 1) {
    LOG(WARNING) << "nbest_size >= 1. Returns empty result.";
    return {};
  }

  if (nbest_size == 1 && !sample) {
    return {Viterbi()};
  }

  // Uses A* search to enumerate N-bests.
  // Given a lattice, enumerates hypotheses (paths) from EOS.
  // At each partial path x, compute f(x) as follows
  //   f(x) = g(x) + h(x).
  // g(x): the sum of scores from  EOS to the left-most node in x.
  //       for a complete hypothesis, g(hyp) is the score of the hypothesis.
  // h(x): a heuristic that estimates the largest score from x to BOS.
  // f(x): the priority to pop a new hypothesis from the priority queue.
  //
  // As left-to-right Viterbi search can tell the *exact* value of h(x),
  // we can obtain the exact n-best results with A*.

  class HypothesisComparator {
  public:
    const bool operator()(Hypothesis *h1, Hypothesis *h2) {
      return (h1->fx < h2->fx);
    }
  };

  using Agenda = std::priority_queue<Hypothesis *,
                                     std::vector<Hypothesis *>,
                                     HypothesisComparator>;
  constexpr size_t kPreallocatedHypothesisSize = 512;
  FreeList<Hypothesis> hypothesis_allocator(kPreallocatedHypothesisSize);

  Agenda agenda;
  std::vector<Lattice::LatticePathWithScore> results;

  auto *eos = hypothesis_allocator.Allocate();
  eos->node = eos_node();
  eos->next = nullptr;
  eos->gx = 0.0;

  std::vector<float> alpha(node_allocator_.size(), 0.0);

  if (sample) {
    // Run forwards algorithm to get normalising constants
    alpha = ForwardAlgorithm(inv_theta);
    // f(eos) = Gumbel(0), as it is the perturbed score of the entire lattice.
    eos->fx = Gumbel();
  } else {
    // Run Viterbi first to fill backtrace score.
    Viterbi();
    eos->fx = eos->node->backtrace_score;
  }
  agenda.push(eos);

  int shrink_count = 0;  // Number of times agenda has shrunk. For logging only.
  bool printed_memory_warning = false;  // For logging only.
  while (!agenda.empty()) {
    auto *top = agenda.top();
    agenda.pop();
    auto *node = top->node;

    // Reaches to BOS
    if (node == bos_node()) {
      results.resize(results.size() + 1);
      for (auto *n = top->next; n->next != nullptr; n = n->next) {
        results.back().first.push_back(n->node);
      }
      results.back().second = top->fx;
      if (results.size() == nbest_size) {
        break;
      }
      continue;
    }

    const int end_nodes_size = end_nodes(node->pos).size();
    std::vector<float> probs(end_nodes_size, 0.0);
    std::vector<float> perturbed_probs(end_nodes_size, 0.0);
    std::vector<double> adjusted_probs(end_nodes_size, 0.0);
    const float Z = alpha[node->node_id];
    if (sample) {
      float max_score = -1e8;
      // Calculate the marginal and perturbed scores for stochastic search
      for (int i = 0; i < end_nodes(node->pos).size(); i++) {
        Node *lnode = end_nodes(node->pos)[i];
        // Calculate backwards transition score
        probs[i] =
            top->gx + alpha[lnode->node_id] + (inv_theta * lnode->score) - Z;
        perturbed_probs[i] = probs[i] + Gumbel();
        if (perturbed_probs[i] > max_score) {
          max_score = perturbed_probs[i];
        }
      }
      // Now constrain the sampled continuations to match the score of parent
      for (int i = 0; i < adjusted_probs.size(); i++) {
        // Use numerically stable version of truncated Gumbel:
        // https://arxiv.org/pdf/1903.06059.pdf appendix B.3
        const float v = top->fx - perturbed_probs[i] +
                        std::log1p(-std::exp(perturbed_probs[i] - max_score));
        adjusted_probs[i] = top->fx - std::max(static_cast<float>(0.0), v) -
                            std::log1p(std::exp(-std::abs(v)));
      }
    }

    // Expands new node ending at node->pos
    for (int i = 0; i < end_nodes(node->pos).size(); i++) {
      Node *lnode = end_nodes(node->pos)[i];
      auto *hyp = hypothesis_allocator.Allocate();
      hyp->node = lnode;
      if (sample) {
        hyp->gx = probs[i];
        hyp->fx = adjusted_probs[i];
      } else {
        hyp->gx = lnode->score + top->gx;  // just adds node->score
        hyp->fx =
            lnode->backtrace_score + top->gx;  // backtrace_score is h(node).
      }
      hyp->next = top;
      agenda.push(hyp);
    }

    static constexpr int kOneBillion = 1000000000;  // 10^9.
    if (hypothesis_allocator.size() >= kOneBillion) {
      if (!printed_memory_warning) {
        printed_memory_warning = true;
        LOG(WARNING) << "Allocator size exceeds " << kOneBillion
                     << " with an example of length " << this->size();
      }
    }

    // When the input is too long or contains duplicated phrases,
    // `agenda` will get extremely big. Here we avoid this case by
    // dynamically shrinking the agenda.
    constexpr int kMaxAgendaSize = 10000;
    constexpr int kMinAgendaSize = 512;
    if (agenda.size() >= kMaxAgendaSize) {
      // Keeps the top `kMinAgendaSize` hypothesis.
      Agenda new_agenda;
      // Keeps the top hypothesis and the ones on their "next" paths.
      FreeList<Hypothesis> new_allocator(kPreallocatedHypothesisSize);
      // Map between old Hypothesis* and new Hypothesis*.
      std::unordered_map<const Hypothesis *, Hypothesis *> clone_map;

      const int size = std::min<int>(kMinAgendaSize, nbest_size * 10);
      shrink_count++;
      LOG(WARNING) << "Too big agenda size " << agenda.size()
                   << ". Shrinking (round " << shrink_count << ") down to "
                   << size << ".";
      for (int i = 0; i < size; ++i) {
        const Hypothesis *top_hyp = agenda.top();
        Hypothesis *cloned_hyp =
            CloneHypAndDependents(top_hyp, &clone_map, &new_allocator);
        new_agenda.push(cloned_hyp);
        agenda.pop();
      }
      agenda = std::move(new_agenda);
      hypothesis_allocator.swap(new_allocator);
    }
  }

  return results;
}

std::vector<Lattice::Node *> Lattice::Sample(float inv_theta) {
  const int len = size();
  if (len == 0) return {};

  std::vector<float> alpha(node_allocator_.size(), 0.0);

  alpha = ForwardAlgorithm(inv_theta);

  auto *mt = GetRandomGenerator();

  std::vector<Node *> results;
  std::vector<float> probs;

  float Z = alpha[eos_node()->node_id];
  Node *node = eos_node();
  while (true) {
    probs.clear();
    for (const Node *lnode : end_nodes_[node->pos]) {
      probs.push_back(std::exp(static_cast<double>(
          alpha[lnode->node_id] + inv_theta * lnode->score - Z)));
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    node = end_nodes_[node->pos][dist(*mt)];
    if (node == bos_node()) break;

    Z = alpha[node->node_id];
    results.push_back(node);
  }

  std::reverse(results.begin(), results.end());
  return results;
}

}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp

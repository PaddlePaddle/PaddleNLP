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
#include <unordered_map>
#include <vector>

#include "utils/shared_mutex.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

static size_t DEFAULT_CACHE_CAPACITY = 10000;
typedef utils::shared_mutex RWLock;
typedef std::unique_lock<RWLock> WLock;
typedef utils::shared_lock<RWLock> RLock;

template <typename K, typename V>
struct Cache {
  std::unordered_map<K, V> map_;
  size_t capacity_;
  Cache(size_t capacity = DEFAULT_CACHE_CAPACITY) : capacity_(capacity) {
    Fresh();
  }

  Cache(const Cache& other) {
    RLock guard(cache_mutex_);
    map_ = other.map_;
    capacity_ = other.capacity_;
  }

  Cache& operator=(const Cache& other) {
    RLock guard(cache_mutex_);
    map_ = other.map_;
    capacity_ = other.capacity_;
    return *this;
  }

  void Fresh() { CreateCacheMap(capacity_); }
  void Clear() {
    WLock guard(cache_mutex_);
    map_.clear();
  }

  bool GetValue(const K& key, V* value) {
    // It's not guaranteed to get the value if the key is in cache
    // for non-blocking read.
    if (cache_mutex_.try_lock_shared()) {
      if (map_.find(key) == map_.end()) {
        cache_mutex_.unlock_shared();
        return false;
      }
      *value = map_.at(key);
      cache_mutex_.unlock_shared();
      return true;
    }
    return false;
  }

  bool SetValue(const K& key, const V& value) {
    // Before trying to acquire a write lock, we check if we are already at
    // capacity with a read handler.
    if (cache_mutex_.try_lock_shared()) {
      if (map_.size() >= capacity_) {
        cache_mutex_.unlock_shared();
        return false;
      }
    } else {
      return false;
    }
    if (cache_mutex_.try_lock()) {
      map_.insert({key, value});
      cache_mutex_.unlock();
      return true;
    }
    return false;
  }

private:
  void CreateCacheMap(size_t capacity) {
    WLock guard(cache_mutex_);
    map_ = std::unordered_map<K, V>(capacity);
  }
  RWLock cache_mutex_;
};

}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp

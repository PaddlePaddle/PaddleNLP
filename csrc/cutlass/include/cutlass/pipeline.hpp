/***************************************************************************************************
 * Copyright (c) 2011-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/numeric/integral_constant.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/gemm/dispatch_policy.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
using namespace arch;
using namespace cute;

// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template<uint32_t Stages_>
struct PipelineState {

  static constexpr uint32_t Stages = Stages_;

private:
  int index_ = 0;
  uint32_t phase_ = 0;

public:
  CUTLASS_DEVICE
  PipelineState(): index_{}, phase_{} {}

  CUTLASS_DEVICE
  PipelineState(int index, uint32_t phase)
    : index_(index)
    , phase_(phase){}

  CUTLASS_DEVICE
  int index() const {
    return index_;
  }

  CUTLASS_DEVICE
  uint32_t phase() const {
    return phase_;
  }

  CUTLASS_DEVICE
  void operator++() {
    ++index_;
    if (index_ == Stages) {
      index_ = 0;
      phase_ ^= 1;
    }
  }

  CUTLASS_DEVICE
  PipelineState& operator=(const PipelineState& other) {
    index_ = other.index();
    phase_ = other.phase();
    return *this;
  }

  CUTLASS_DEVICE
  PipelineState advance(uint32_t num_iterations) {
    // Number of iterations cross over the stage boundary => flipped phase
    if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages ) {
      phase_ ^= 1;
    }
    // How many times number of iterations cross over the stage boundary and
    // end up on a odd number => flipped phase
    if ((num_iterations >= Stages) && (((index_ + num_iterations) / Stages) % 2) == 1) {
      phase_ ^= 1;
    }
    index_ = (index_ + num_iterations) % Stages;
    return *this;
  }

  CUTLASS_DEVICE
  static PipelineState make_pipeline_state(PipelineState start_state, uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};

template<class Pipeline>  
CUTLASS_DEVICE
PipelineState<Pipeline::Stages> make_producer_start_state()
{
  // Producer starts with an opposite phase as the buffer are initially empty
  constexpr int InitialProducerStage = 0;
  constexpr uint32_t InitialProducerPhase = 1;
  return {InitialProducerStage, InitialProducerPhase};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA (producer) Async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////
// Assumptions : Constructor is Visible Cluster-wide (as it needs a Cluster-Sync)
// We have exactly one thread elected in the Producer as the "leader"
// Currently, it is optional to elect a leader for the Consumers
template <int Stages_, class ClusterShape_>
class PipelineTmaAsync {
public :
  using ClusterShape = ClusterShape_;
  using FullBarrier = ClusterTransactionBarrier;
  using EmptyBarrier = ClusterBarrier;
  using ValueType = FullBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    uint32_t transaction_bytes = 0;
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0;
  };

private :
  //
  // Data Members
  //
  uint32_t dst_blockid_ = 0;
  uint32_t is_signalling_thread_ = 0;
  FullBarrier *full_barrier_ptr_ = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;
  Params params_;

  //
  // Methods
  //

public:
  // Constructor
  CUTLASS_DEVICE
  PipelineTmaAsync(SharedStorage& storage, Params params)
      : params_(params)
      , full_barrier_ptr_(&storage.full_barrier_[0])
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();
    auto cluster_shape = ClusterShape{};

    if (warp_idx == 0 && lane_predicate == 1) {
      // Barrier FULL init
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr_[i].init(1);
      }

      // Barrier EMPTY init
      uint32_t const num_consumers = cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1;
      for (int i = 0; i < Stages; ++i) {
        empty_barrier_ptr_[i].init(num_consumers);
      }
    }

    // Logic to optimally schedule Empty Arrives
    // Goal : To divide SYNCS Empty Arrival duty equally amongst the Warp-Group (128 threads)
    dim3 block_id = block_id_in_cluster();
    auto cluster_size = cute::size(cluster_shape);
    static constexpr int MaxClusterSize = 16;
    static_assert(cluster_size <= MaxClusterSize, "ERROR : Cluster size too large !" );

    // STEP 1 : Use Cute Layout function to generate an optimal dst block-id (0-15)
    if (params_.num_consumers == 128) {
      int thread_idx = threadIdx.x % 128;
      is_signalling_thread_ = (thread_idx % (128 / MaxClusterSize)) == 0;
      auto layout = cute::composition(Swizzle<2,0,-2>{},
                                Layout<Shape<_4,_4>,Stride<_4, _1>>{});
      uint32_t thread_row = warp_idx % 4;
      uint32_t thread_col = (thread_idx / 8) % 4;
      dst_blockid_ = layout(thread_row, thread_col);
    }
    else if (params_.num_consumers == 32){
      int thread_idx = threadIdx.x % 32;
      is_signalling_thread_ = (thread_idx % (32 / MaxClusterSize)) == 0;
      auto layout = Layout<Shape<_4,_4>,Stride<_4, _1>>{};
      uint32_t thread_row = thread_idx / 8;
      uint32_t thread_col = (thread_idx % 8) / 2;
      dst_blockid_ = layout(thread_row, thread_col);
    }
    else {
      is_signalling_thread_ = 0;
    }

    // STEP 2: Find if this dst block-id needs an arrival for this problem
    is_signalling_thread_ &= dst_blockid_ < cluster_size;
    is_signalling_thread_ &= is_same_row_or_col(dst_blockid_, block_id, cluster_shape);

    cutlass::arch::fence_barrier_init();
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait = false) {
    // 1. Wait for empty barrier to be ready
    // 2. Set the transaction bytes set to occur on the Full barrier
    uint32_t done = empty_barrier_ptr_[stage].test_wait(phase, (!skip_wait));
    if ((!done) && (!skip_wait)){
      empty_barrier_ptr_[stage].wait(phase);
    }

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_reset_bytes(params_.transaction_bytes);
    }

  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState<Stages> state) {
    producer_acquire(state.index(), state.phase());
  }

  // NOP for TMA based mainloop
  CUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t bytes) {
    // Below code is used only for unit-testing (in the absennce of TMA commit)
    #if CUTLASS_UNIT_TEST_PIPELINE
      if (params_.is_leader) {
        // STEP 1 : Commit to self
        full_barrier_ptr_[stage].commit(bytes);

        // STEP 2 : Commit to other blocks in our cluster
        auto cluster_shape = ClusterShape{};
        Layout block_layout_in_cluster = make_layout(cluster_shape);
        dim3 local_block_id = cute::block_id_in_cluster();

        CUTLASS_PRAGMA_UNROLL
        for(int n = 0; n < size<1>(block_layout_in_cluster); ++n) {
          uint32_t dst_block_id = block_layout_in_cluster(local_block_id.x,n,Int<0>{});
          full_barrier_ptr_[stage].commit(dst_block_id, bytes, n!=local_block_id.y);
        }

        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < size<0>(block_layout_in_cluster); ++m) {
          uint32_t dst_block_id = block_layout_in_cluster(m,local_block_id.y,Int<0>{});
          full_barrier_ptr_[stage].commit(dst_block_id, bytes, m!=local_block_id.x);
        }
      }
    #endif
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState<Stages> state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }


  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    uint32_t done = full_barrier_ptr_[stage].test_wait(phase);
    if (!done){
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState<Stages> state) {
    consumer_wait(state.index(), state.phase());
  }

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signalling_thread_ & (!skip));
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState<Stages> state) {
    consumer_release(state.index());
  }

  CUTLASS_DEVICE
  ValueType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ValueType*>(&full_barrier_ptr_[stage]);
  }

  CUTLASS_DEVICE
  bool is_same_row_or_col(int dst_block_id, dim3 block_id, ClusterShape cluster_shape) {
    return ((dst_block_id % cute::size<0>(cluster_shape)) == block_id.x ||
            (dst_block_id / cute::size<0>(cluster_shape)) == block_id.y);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple producer-consumer async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// *Count Signifies the number of producers / consumers who will announce their completion

template <int Stages_>
class PipelineAsync {
public :
  using FullBarrier = ClusterBarrier;
  using EmptyBarrier = ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
    uint32_t dst_blockid = cute::block_rank_in_cluster();
  };

private:
  //
  // Data Members
  //
  Params params_;
  FullBarrier *full_barrier_ptr_;
  EmptyBarrier *empty_barrier_ptr_;

public:

  // Default assumption when only storage is passed is :
  // => single producer, single consumer & they are in the same block (within the Cluster)
  CUTLASS_DEVICE
  PipelineAsync(SharedStorage& storage)
    : PipelineAsync(storage, {}) {}

  CUTLASS_DEVICE
  PipelineAsync(
    SharedStorage& storage,
    Params const& params) :
      params_(params),
      full_barrier_ptr_(&storage.full_barrier_[0]),
      empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();

    // Barrier FULL, EMPTY init
    // Init is done only by thread 0 of the block
    if (warp_idx == 0 && lane_predicate == 1) {
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr_[i].init(params.producer_arv_count);
        empty_barrier_ptr_[i].init(params.consumer_arv_count);
      }
    }

    cutlass::arch::fence_barrier_init();
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait = false) {
    uint32_t done = empty_barrier_ptr_[stage].test_wait(phase, (!skip_wait));
    if ((!done) && (!skip_wait)){
      empty_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState<Stages> state) {
    producer_acquire(state.index(), state.phase());
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    full_barrier_ptr_[stage].arrive();
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState<Stages> state) {
    producer_commit(state.index());
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    uint32_t done = full_barrier_ptr_[stage].test_wait(phase);
    if (!done){
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState<Stages> state) {
    consumer_wait(state.index(), state.phase());
  }

  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive(params_.dst_blockid, (not skip));
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState<Stages> state) {
    consumer_release(state.index());
  }

  CUTLASS_DEVICE
  ProducerBarrierType* get_producer_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }
};



///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Barrier to ensure an Ordered Sequence between
// SequenceLength number of groups (each with group_size participants) executing SequenceDepth Stages
// i.e., for all i < j - only after id "i" arrives at a particular stage "m"
// will the wait() for id "j" succeed for the same stage
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template<int SequenceDepth, int SequenceLength>
class OrderedSequenceBarrier {
public :
  using Barrier = ClusterBarrier;

  struct SharedStorage {
    Barrier barrier_[SequenceDepth][SequenceLength];
  };

  struct Params {
    uint32_t group_id;
    uint32_t group_size;
  };

private :
  //
  // Data Members
  //

  // In future this Params object can be replaced easily with a CG object
  Params params_;
  Barrier *barrier_ptr_;
  PipelineState<SequenceDepth> stage_;

  static constexpr int Depth = SequenceDepth;
  static constexpr int Length = SequenceLength;

public:
  OrderedSequenceBarrier() = delete;
  OrderedSequenceBarrier(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier(OrderedSequenceBarrier&&) = delete;
  OrderedSequenceBarrier& operator=(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier& operator=(OrderedSequenceBarrier&&) = delete;
  ~OrderedSequenceBarrier() = default;

  CUTLASS_DEVICE
  OrderedSequenceBarrier(SharedStorage& storage, Params const& params) :
      params_(params),
      barrier_ptr_(&storage.barrier_[0][0]),
      // Group 0 - starts with an opposite phase
      stage_({0, params.group_id == 0}) {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();

    // Barrier FULL, EMPTY init
    // Init is done only by the one elected thread of the block
    if (warp_idx == 0 && lane_predicate == 1) {
      for (int d = 0; d < Depth; ++d) {
        for (int l = 0; l < Length; ++l) {
          barrier_ptr_[d * Length + l].init(params.group_size);
        }
      }
    }

    cutlass::arch::fence_barrier_init();
  }

  // Wait on a stage to be unlocked
  CUTLASS_DEVICE
  void wait() {
    get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
  }

  // Signal completion of Stage and move to the next stage
  // (group_id) signals to (group_id+1)
  CUTLASS_DEVICE
  void arrive() {
    int signalling_id = (params_.group_id + 1) % Length;
    get_barrier_for_current_stage(signalling_id).arrive();
    ++stage_;
  }

private:

  CUTLASS_DEVICE
  Barrier& get_barrier_for_current_stage(int group_id) {
    return barrier_ptr_[stage_.index() * Length + group_id];
  }
};

}  // end namespace cutlass

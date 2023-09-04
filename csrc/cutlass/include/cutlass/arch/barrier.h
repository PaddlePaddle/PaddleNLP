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
/*! \file
    \brief Barrier Operations on SM90+
*/

#pragma once

#include <cutlass/arch/memory_sm75.h>
#include <cute/arch/cluster_sm90.hpp>

namespace cutlass {
/// @brief
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && (__CUDACC_VER_MAJOR__ >= 12)
#define CUDA_BARRIER_ENABLED 1
#else
#define CUDA_BARRIER_ENABLED 0
#endif

class NamedBarrier {

  // Data Members:

  // Range = [1 , NUM_THREADS_PER_CTA]
  // Range % warp-size (i.e 32) == 0
  uint32_t const num_threads_;

  // Range : [0, 15]
  uint32_t const id_;

 public:

  CUTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, uint32_t id = 0)
      : num_threads_(num_threads), id_(id) {}

  CUTLASS_DEVICE 
  void arrive_and_wait() const { 
    NamedBarrier::arrive_and_wait(num_threads_, id_); 
  }

  CUTLASS_DEVICE
  void arrive() const {
    NamedBarrier::arrive(num_threads_, id_);
  }

  CUTLASS_DEVICE 
  void sync() const { 
    NamedBarrier::arrive_and_wait(); 
  }

  //  Static variants
  CUTLASS_DEVICE 
  static void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void sync(uint32_t num_threads, uint32_t barrier_id) {
    NamedBarrier::arrive_and_wait(num_threads, barrier_id);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper introduces a new cluster-wide barrier which handle with Cluster-wide AW behaviour.
// This is an extension to the Ampere AW barriers
// Note : Ampere AW Barriers have a larger max-arrive count (2^30) than Hopper AW Barriers (2^20).
struct ClusterBarrier {

  using ValueType = uint64_t;

protected:
  // Can never be initializated - can only be aliased to smem
  ValueType barrier_;

public:

  CUTLASS_DEVICE
  ClusterBarrier() = delete;

  CUTLASS_DEVICE 
  void init(uint32_t arrive_count) const {
    ClusterBarrier::init(&this->barrier_, arrive_count);
  }

  CUTLASS_DEVICE
  uint32_t test_wait(uint32_t phase, uint32_t pred=true) const {
    return ClusterBarrier::test_wait(&this->barrier_, phase, pred);
  }

  CUTLASS_DEVICE
  void wait(uint32_t phase) const {
    ClusterBarrier::wait(&this->barrier_, phase);
  }

  // Barrier arrive on local smem
  CUTLASS_DEVICE
  void arrive() const {
    ClusterBarrier::arrive(&this->barrier_);
  }

  // Remote SMEM arrive with a perdicate (usually done to pick the thread doing the arrive)
  CUTLASS_DEVICE
  void arrive(uint32_t cta_id, uint32_t pred = true ) const {
    ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
  }

  //
  //  Static Versions
  //
  CUTLASS_DEVICE
  static void init(ValueType const* smem_ptr, uint32_t arrive_count) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.init.shared.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Static version of wait - in case we don't want to burn a register
  CUTLASS_DEVICE
  static void wait(ValueType const* smem_ptr, uint32_t phase) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra.uni DONE; \n\t"
        "bra.uni     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks));

#else
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static uint32_t test_wait(ValueType const* smem_ptr, uint32_t phase, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        ".reg .pred P2; \n\t"
        "setp.eq.u32 P2, %3, 1;\n\t"
        "@P2 mbarrier.test_wait.parity.shared.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase), "r"(pred));

    return waitComplete;
#else
    asm volatile ("brkpt;\n" ::);
#endif
    return 0;
  }

  // Static Predicated version of the above - in case we know the address.
  CUTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Barrier arrive on local smem
  CUTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint64_t state = 0;
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.shared.b64 %1, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr), "l"(state));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void invalidate(ValueType const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.ival.shared.b64 [%0]; \n\t"
        "}"
        :
        : "r"(smem_addr));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SM90 also introduces a new type of cluster-barrier which supports sync.
// not just based on Arrive Count, but also transaction count (in bytes)
struct ClusterTransactionBarrier : public ClusterBarrier {

  CUTLASS_DEVICE
  ClusterTransactionBarrier() = delete;

  // Performs an arrive operation + bytes reset
  CUTLASS_DEVICE
  void arrive_and_reset_bytes(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::arrive_and_reset_bytes(&this->barrier_, transaction_bytes);
  }

  // Performs an arrive operation + bytes reset
  CUTLASS_DEVICE
  void arrive_and_reset_bytes(uint32_t transaction_bytes, uint32_t cta_id) const {
    ClusterTransactionBarrier::arrive_and_reset_bytes(&this->barrier_, transaction_bytes , cta_id, true);
  }

  CUTLASS_DEVICE
  void commit(uint32_t transaction_bytes, uint32_t pred = 1) const {
    uint32_t cta_rank = cute::block_rank_in_cluster();
    ClusterTransactionBarrier::commit(&this->barrier_, cta_rank, transaction_bytes, pred);
  }

  CUTLASS_DEVICE
  void commit(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
    ClusterTransactionBarrier::commit(&this->barrier_, dst_cta_id, transaction_bytes, pred);
  }

  //
  //  Static Versions
  //

  // Performs an arrive operation + bytes reset
  CUTLASS_DEVICE
  static void arrive_and_reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Performs an arrive operation + bytes reset for a remote cta_id in a Cluster
  CUTLASS_DEVICE
  static void arrive_and_reset_bytes(
      ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Performs an bytes reset without doing an arrive operation
  CUTLASS_DEVICE
  static void reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.expect_tx.shared.b64 [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }

  // Increments transaction bytes in the barrier
  CUTLASS_DEVICE
  static void commit(
      ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    smem_addr = cute::set_block_rank(smem_addr, dst_cta_id);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64   [%1], %0;"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr), "r"(pred));
#else
    asm volatile ("brkpt;\n" ::);
#endif
  }
};

// Helps with visibility of barrier init operations across warps / cta / cluster
// Available as a separate function so as to batch inits across barriers and fence once
// Note : It must be composed with an appropriate sync instruction with the right scope
// to ensure visibility eg. __syncthreads() or a cluster_arrive() + cluster_wait()
CUTLASS_DEVICE
void fence_barrier_init() {
#if CUDA_BARRIER_ENABLED
  asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}"
      ::);
#else
    asm volatile ("brkpt;\n" ::);
#endif
}

// Issue a shared memory fence for async operations
CUTLASS_DEVICE
void fence_view_async_shared() {
#if CUDA_BARRIER_ENABLED
    asm volatile (
        "{\n\t"
        "fence.proxy.async.shared::cta; \n"
        "}"
        ::);
#else
    asm volatile ("brkpt;\n" ::);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // end namespace arch
}  // end namespace cutlass

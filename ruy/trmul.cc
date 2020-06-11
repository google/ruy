/* Copyright 2019 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ruy/trmul.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "ruy/allocator.h"
#include "ruy/block_map.h"
#include "ruy/check_macros.h"
#include "ruy/common.h"
#include "ruy/cpu_cache_params.h"
#include "ruy/cpuinfo.h"
#include "ruy/ctx.h"
#include "ruy/mat.h"
#include "ruy/matrix.h"
#include "ruy/mul_params.h"
#include "ruy/opt_set.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/side_pair.h"
#include "ruy/size_util.h"
#include "ruy/thread_pool.h"
#include "ruy/tune.h"

namespace ruy {

namespace {

// Enum to track the packingstatus of a block of the LHS or RHS matrix.
enum class PackingStatus : std::uint8_t {
  kNotStarted,  // No thread has started packing this block yet.
  kInProgress,  // Some thread is currently packing this block.
  kFinished     // This block has already been packed.
};

// TrMulTask is the task that a ruy thread runs to perform the TrMul operation.
struct TrMulTask final : Task {
  TrMulTask(TrMulParams* params_, const BlockMap& block_map_,
            std::atomic<int>* atomic_block_id_, int thread_id_,
            bool need_atomics_,
            SidePair<std::atomic<PackingStatus>*> packing_status_,
            TuningResolver* tuning_resolver_, Allocator* local_allocator_)
      : params(params_),
        block_map(block_map_),
        atomic_block_id(atomic_block_id_),
        thread_id(thread_id_),
        need_atomics(need_atomics_),
        packing_status(packing_status_),
        tuning_resolver(tuning_resolver_),
        local_allocator(local_allocator_),
        local_packed{nullptr, nullptr} {}

  // Thread main function. This is one thread's share of the TrMul work.
  void Run() override {
    // Allocate and initialize `local_packed`.
    for (Side side : {Side::kLhs, Side::kRhs}) {
      if (!params->is_prepacked[side]) {
        const int size = NumBlocksPerSide(side, block_map);
        local_allocator->Allocate(size, &local_packed[side]);
        memset(local_packed[side], 0, size * sizeof(bool));
      }
    }

    const Tuning tuning = tuning_resolver->Resolve();
    const int num_blocks = NumBlocks(block_map);

    // Each thread starts by initially reserving the block whose id
    // is the thread id.
    int block_id = thread_id;
    // Loop until all blocks have been computed.
    while (block_id < num_blocks) {
      // Reserve the next block to handle, hiding the latency of this atomic op.
      const int next_block_id =
          atomic_block_id->fetch_add(1, std::memory_order_relaxed);
      // Get coordinates of the current block to handle, in "block space".
      SidePair<int> block;
      GetBlockByIndex(block_map, block_id, &block);
      // Get coordinates of the current block to handle, in matrix space.
      SidePair<int> start, end;
      GetBlockMatrixCoords(block_map, block, &start, &end);
      // Maybe pack the current LHS/RHS block, if not already packed.
      EnsurePacked(block, start, end, tuning);
      // Actually do matrix multiplication work
      params->RunKernel(tuning, start, end);
      // Move on to the next block as obtained by the atomic increment
      // at the start of this while loop iteration.
      block_id = next_block_id;
    }

    local_allocator->FreeAll();
  }

 private:
  // Tries to pack a block, without blocking.
  // If the block was already packed, returns true.
  // If the block was not started packing, packs it and returns true.
  // If the block was being packed by another thread, returns false.
  bool TryPack(Side side, int block, int start, int end, Tuning tuning) {
    if (params->is_prepacked[side]) {
      return true;
    }
    if (!local_packed[side][block]) {
      if (need_atomics) {
        // Explanation of this compare_exchange_strong operation:
        // This atomically performs all of the following:
        // 1. Read `status` with "acquire" memory order.
        //    * That this read uses "acquire" is because both memory orders
        //      specified have "acquire" as their read-component.
        // 2. Compare (bitwise) with `exchanged_status`.
        // 3. If equal, stores the value kInProgress to `status` with "release"
        //    memory order, and returns true, so we take this 'if' branch.
        //    * That this store uses "release" is because of the _rel part in
        //      memory_order_acq_rel passed as the first memory order argument.
        // 4. If not equal, stores the loaded value of `status` to
        //    `exchanged_status` with "relaxed" semantics, and returns false,
        //    so we take the 'else' branch.
        //    * That this store uses "relaxed" is because the second memory
        //      order argument, memory_order_acquire, implies no particular
        //      store semantics. "relaxed" is acceptable here because this
        //      stores to a local stack variable.
        //
        // Rationale for compare_exchange_strong as opposed to
        // compare_exchange_weak:
        // The spurious-failure case with compare_exchange_weak will actually
        // happen a lot here, because the atomic 'status' bytes are stored
        // contiguously in arrays and neighboring values will be accessed
        // by multiple threads concurrently. On a typical ARM CPU, an exclusives
        // reservation granule is 64 bytes, so a lot of false-sharing may
        // happen. Using compare_exchange_weak would thus result in often having
        // TryPack return 'false' when it could instead have done the packing
        // work and returned 'true'. Heuristically, that is not a good thing.
        // Moreover, this changes the TryPack contract, loosening it and making
        // it harder for the caller to reason about. Finally, the overhead of
        // atomic operations is mitigated by the enclosing check on
        // local_packed, so maybe the overhead of compare_exchange_strong isn't
        // such a problem. But we don't really know for sure, that would be
        // interesting to experiment more with.
        PackingStatus exchanged_status = PackingStatus::kNotStarted;
        std::atomic<PackingStatus>& status = packing_status[side][block];
        if (status.compare_exchange_strong(
                exchanged_status, PackingStatus::kInProgress,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
          // In this branch, the status was kNotStarted and we just atomically
          // changed it to kInProgress as we are about to handle the packing
          // ourselves.
          params->RunPack(side, tuning, start, end);
          status.store(PackingStatus::kFinished, std::memory_order_release);
        } else if (exchanged_status == PackingStatus::kInProgress) {
          // Another thread is currently packing this block.
          return false;
        }
        RUY_DCHECK(status.load(std::memory_order_acquire) ==
                   PackingStatus::kFinished);
      } else {
        // Single-threaded case: no need for expensive atomics, local_packed
        // is the truth already.
        params->RunPack(side, tuning, start, end);
      }
      local_packed[side][block] = true;
    }
    return true;
  }

  // Ensures that both the LHS and RHS blocks required by the specified block
  // are packed. In the event that they are already being packed on another
  // threads, this function may perform the packing of some other block while
  // waiting for that other thread to finish packing the requested block.
  void EnsurePacked(const SidePair<int>& block, const SidePair<int>& start,
                    const SidePair<int>& end, Tuning tuning) {
#if RUY_OPT(PACK_AHEAD)
    SidePair<int> next_runahead_block{block[Side::kLhs] + 1,
                                      block[Side::kRhs] + 1};
    Side next_runahead_side = Side::kLhs;
#endif
    while (true) {
      bool both_sides_packed = true;
      for (Side side : {Side::kLhs, Side::kRhs}) {
        both_sides_packed &=
            TryPack(side, block[side], start[side], end[side], tuning);
      }
      if (both_sides_packed) {
        break;
      }
#if RUY_OPT(PACK_AHEAD)
      const Side runahead_side = next_runahead_side;
      const int runahead_block = next_runahead_block[runahead_side];
      next_runahead_side =
          next_runahead_side == Side::kLhs ? Side::kRhs : Side::kLhs;
      if (runahead_block >= NumBlocksPerSide(runahead_side, block_map)) {
        continue;
      }
      int runahead_block_start, runahead_block_end;
      GetBlockMatrixCoords(runahead_side, block_map, runahead_block,
                           &runahead_block_start, &runahead_block_end);
      TryPack(runahead_side, runahead_block, runahead_block_start,
              runahead_block_end, tuning);
      next_runahead_block[runahead_side] = runahead_block + 1;
#endif
    }
  }

  TrMulParams* params;
  const BlockMap& block_map;
  std::atomic<int>* atomic_block_id;
  int thread_id;
  bool need_atomics;
  SidePair<std::atomic<PackingStatus>*> packing_status;
  TuningResolver* tuning_resolver;
  Allocator* local_allocator;

  // Local indicators of packedness to avoid the overhead of atomic ops.
  SidePair<bool*> local_packed;
};

void AllocatePMatrix(Allocator* allocator, PEMat* packed) {
  packed->data = allocator->AllocateBytes(DataBytes(*packed));
  packed->sums = allocator->AllocateBytes(SumsBytes(*packed));
}

int GetThreadCount(Ctx* ctx, int rows, int cols, int depth) {
#if RUY_PLATFORM_EMSCRIPTEN
  // b/139927184, std::thread constructor raises exception
  return 1;
#endif
  // Empirically determined rule for reasonable number of
  // threads to use. This is proportional to the number of arithmetic ops
  // in this Mul (product of the 3 sizes).
  static constexpr int kDivisorLog2 = 15;
  const int guess_log2 = std::max(
      0, ceil_log2(rows) + ceil_log2(cols) + ceil_log2(depth) - kDivisorLog2);
  return std::min(1 << guess_log2, ctx->max_num_threads());
}

LoopStructure GetLoopStructure(int tentative_thread_count, int rows, int cols,
                               int depth, int lhs_scalar_size,
                               int rhs_scalar_size,
                               const CpuCacheParams& cpu_cache_params) {
  if (tentative_thread_count == 1) {
    const BlockMapTraversalOrder traversal_order = GetTraversalOrder(
        rows, cols, depth, lhs_scalar_size, rhs_scalar_size, cpu_cache_params);
    // If we are in the GEMV case or the block_map would be using linear
    // traversal anyway, use the simple loop.
    if ((cols == 1) || traversal_order == BlockMapTraversalOrder::kLinear) {
      return LoopStructure::kSimple;
    }
  }
  return LoopStructure::kGeneral;
}

}  // namespace

void TrMul(TrMulParams* params, Ctx* ctx) {
  profiler::ScopeLabel label(
      "TrMul (Path=0x%x, max_num_threads=%d, is_prepacked=(%d,%d))",
      static_cast<int>(params->path), ctx->max_num_threads(),
      params->is_prepacked[Side::kLhs], params->is_prepacked[Side::kRhs]);

  PEMat& packed_lhs = params->packed[Side::kLhs];
  PEMat& packed_rhs = params->packed[Side::kRhs];
  EMat& lhs = params->src[Side::kLhs];
  EMat& rhs = params->src[Side::kRhs];

  const int rows = lhs.layout.cols;
  const int cols = rhs.layout.cols;
  const int depth = lhs.layout.rows;

  const int tentative_thread_count = GetThreadCount(ctx, rows, cols, depth);
  const auto& cpu_cache_params = ctx->mutable_cpuinfo()->CacheParams();
  const auto loop_structure = GetLoopStructure(
      tentative_thread_count, rows, cols, depth, lhs.data_type.size,
      rhs.data_type.size, cpu_cache_params);
  Allocator* allocator = ctx->GetMainAllocator();

  // Allocate packed matrices
  for (Side side : {Side::kLhs, Side::kRhs}) {
    if (!params->is_prepacked[side]) {
      AllocatePMatrix(allocator, &params->packed[side]);
    }
  }

  // Case of running this TrMul as a simple loop.
  // This is a good place to start reading this function: all the rest
  // of this function is just an optimized, but functionally equivalent,
  // version of that.
  if (loop_structure == LoopStructure::kSimple) {
    profiler::ScopeLabel label_simple("TrMulImpl, simple loop");
    Tuning tuning = ctx->GetMainThreadTuning();

    const SidePair<int> origin{0, 0};
    const SidePair<int> rounded_dims{packed_lhs.layout.cols,
                                     packed_rhs.layout.cols};
    for (Side side : {Side::kLhs, Side::kRhs}) {
      if (!params->is_prepacked[side]) {
        params->RunPack(side, tuning, origin[side], rounded_dims[side]);
      }
    }
    params->RunKernel(tuning, origin, rounded_dims);

    allocator->FreeAll();
    return;
  }

  profiler::ScopeLabel label_general("TrMulImpl, general case");

  // Initialize block map.
  BlockMap block_map;
  MakeBlockMap(packed_lhs.layout.cols, packed_rhs.layout.cols, depth,
               packed_lhs.layout.kernel.cols, packed_rhs.layout.kernel.cols,
               packed_lhs.data_type.size, packed_rhs.data_type.size,
               tentative_thread_count, cpu_cache_params, &block_map);

  // Initialize per-thread state.
  const int thread_count = block_map.thread_count;
  const bool need_atomics = thread_count > 1;
  ctx->EnsureThreadSpecificResources(thread_count);
  for (int i = 0; i < thread_count; i++) {
    ctx->GetThreadSpecificTuningResolver(i)->SetTuning(ctx->explicit_tuning());
  }

  // In the need_atomics case, allocate and initialize atomic values tracking
  // the packing status of blocks.
  SidePair<std::atomic<PackingStatus>*> packing_status{nullptr, nullptr};
  if (need_atomics) {
    for (Side side : {Side::kLhs, Side::kRhs}) {
      if (!params->is_prepacked[side]) {
        const int size = NumBlocksPerSide(side, block_map);
        allocator->Allocate(size, &packing_status[side]);
        for (int i = 0; i < size; i++) {
          packing_status[side][i].store(PackingStatus::kNotStarted,
                                        std::memory_order_relaxed);
        }
      }
    }
  }

  // Create the atomic block id, allocate it using Allocator so that
  // we get the alignment ensuring that it sits alone in its exclusives
  // reservation granule.
  std::atomic<int>* atomic_block_id;
  allocator->Allocate(1, &atomic_block_id);

  // Create task objects.
  TrMulTask* tasks;
  allocator->Allocate(thread_count, &tasks);

  atomic_block_id->store(thread_count);

  for (int i = 0; i < thread_count; i++) {
    auto* allocator = ctx->GetThreadSpecificAllocator(i);
    auto* tuning_resolver = ctx->GetThreadSpecificTuningResolver(i);
    new (tasks + i)
        TrMulTask(params, block_map, atomic_block_id, i, need_atomics,
                  packing_status, tuning_resolver, allocator);
  }

  // Do the computation.
  ctx->mutable_thread_pool()->Execute(thread_count, tasks);

  // Finish up.
  for (int i = 0; i < thread_count; i++) {
    tasks[i].~TrMulTask();
  }

  allocator->FreeAll();
}

}  // namespace ruy

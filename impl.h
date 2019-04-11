// Gory implementation layer.  No documentation so far, come back in a few
// days...

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_IMPL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_IMPL_H_

#include <vector>

#include "third_party/gemmlowp/profiling/instrumentation.h"
#include "allocator.h"
#include "block_map.h"
#include "common.h"
#include "context.h"
#include "kernel.h"
#include "opt_set.h"
#include "pack.h"
#include "thread_pool.h"
#include "trace.h"
#include "tune.h"

namespace ruy {

template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename PackedLhsScalar, typename PackedRhsScalar,
          typename DstScalar, typename Spec>
struct TrMulTask final : Task {
  using AccumScalar = typename Spec::AccumScalar;
  TrMulTask(const Matrix<LhsScalar>& lhs_, const Matrix<RhsScalar>& rhs_,
            Matrix<PackedLhsScalar>* packed_lhs_,
            Matrix<PackedRhsScalar>* packed_rhs_, Matrix<DstScalar>* result_,
            const BlockMap& block_map_,

            std::atomic<std::uint32_t>* atomic_n_, std::uint32_t thread_id_,
            std::atomic<bool>* lhs_packed_, std::atomic<bool>* rhs_packed_,
            const Spec& spec_, TuningResolver* tuning_resolver_,
            Allocator* local_allocator_, Trace* trace_)
      : lhs(lhs_),
        rhs(rhs_),
        packed_lhs(packed_lhs_),
        packed_rhs(packed_rhs_),
        result(result_),
        block_map(block_map_),
        atomic_n(atomic_n_),
        thread_id(thread_id_),
        lhs_packed(lhs_packed_),
        rhs_packed(rhs_packed_),
        spec(spec_),
        tuning_resolver(tuning_resolver_),
        local_allocator(local_allocator_),
        trace(trace_) {}

  void Run() override {
    TraceRecordThreadStart(thread_id, trace);

    std::uint16_t num_blocks_of_rows = NumBlocksOfRows(block_map);
    std::uint16_t num_blocks_of_cols = NumBlocksOfCols(block_map);
    std::uint32_t num_blocks = NumBlocks(block_map);

    Allocator::Handle<bool> local_lhs_packed_handle;
    Allocator::Handle<bool> local_rhs_packed_handle;

    if (lhs_packed) {
      Reserve(local_allocator, num_blocks_of_rows, &local_lhs_packed_handle);
    }
    if (rhs_packed) {
      Reserve(local_allocator, num_blocks_of_cols, &local_rhs_packed_handle);
    }

    local_allocator->Commit();

    bool* local_lhs_packed = nullptr;
    bool* local_rhs_packed = nullptr;

    if (lhs_packed) {
      local_lhs_packed = local_allocator->GetPointer(local_lhs_packed_handle);
      memset(local_lhs_packed, 0, num_blocks_of_rows * sizeof(bool));
    }
    if (rhs_packed) {
      local_rhs_packed = local_allocator->GetPointer(local_rhs_packed_handle);
      memset(local_rhs_packed, 0, num_blocks_of_cols * sizeof(bool));
    }

    using Kernel =
        Kernel<ThePath, PackedLhsScalar, PackedRhsScalar, DstScalar, Spec>;
    using LhsKernelLayout = typename Kernel::RhsLayout;
    using RhsKernelLayout = typename Kernel::RhsLayout;

    const Tuning tuning = tuning_resolver->Resolve();
    Kernel kernel(tuning);

    TraceRecordThreadLoopStart(thread_id, trace);

    std::uint16_t block_r, block_c;
    int start_r, start_c, end_r, end_c;
    std::uint16_t next_block_r, next_block_c;
    int next_start_r, next_start_c, next_end_r, next_end_c;

    std::uint32_t n = thread_id;
    std::uint32_t next_n;

    GetBlockByIndex(block_map, n, &block_r, &block_c);
    TraceRecordBlockReserved(thread_id, n, trace);
    GetBlockMatrixCoords(block_map, block_r, block_c, &start_r, &start_c,
                         &end_r, &end_c);
    TraceRecordBlockCoordsComputed(n, trace);
    while (n < num_blocks) {
      next_n = atomic_n->fetch_add(1, std::memory_order_relaxed);
      if (next_n < num_blocks) {
        GetBlockByIndex(block_map, next_n, &next_block_r, &next_block_c);
        TraceRecordBlockReserved(thread_id, next_n, trace);
        GetBlockMatrixCoords(block_map, next_block_r, next_block_c,
                             &next_start_r, &next_start_c, &next_end_r,
                             &next_end_c);
        TraceRecordBlockCoordsComputed(next_n, trace);
      }
      if (start_r < end_r && start_c < end_c) {
        if (local_lhs_packed && !local_lhs_packed[block_r]) {
          if (!lhs_packed[block_r].load(std::memory_order_acquire)) {
            Pack<ThePath, LhsKernelLayout>(tuning, lhs, packed_lhs, start_r,
                                           end_r);
            TraceRecordBlockPackedLhs(n, trace);
            local_lhs_packed[block_r] = true;
            lhs_packed[block_r].store(true, std::memory_order_release);
          }
        }
        if (local_rhs_packed && !local_rhs_packed[block_c]) {
          if (!rhs_packed[block_c].load(std::memory_order_acquire)) {
            Pack<ThePath, RhsKernelLayout>(tuning, rhs, packed_rhs, start_c,
                                           end_c);
            TraceRecordBlockPackedRhs(n, trace);
            local_rhs_packed[block_c] = true;
            rhs_packed[block_c].store(true, std::memory_order_release);
          }
        }
        RunKernel(kernel, *packed_lhs, *packed_rhs, spec, start_r, start_c,
                  end_r, end_c, result);
      }
      TraceRecordBlockFinished(n, trace);
      n = next_n;
      block_r = next_block_r;
      block_c = next_block_c;
      start_r = next_start_r;
      start_c = next_start_c;
      end_r = next_end_r;
      end_c = next_end_c;
    }

    local_allocator->Decommit();

    TraceRecordThreadEnd(thread_id, trace);
  }

 private:
  const Matrix<LhsScalar>& lhs;
  const Matrix<RhsScalar>& rhs;
  Matrix<PackedLhsScalar>* packed_lhs;
  Matrix<PackedRhsScalar>* packed_rhs;

  Matrix<DstScalar>* result;
  const BlockMap& block_map;
  std::atomic<std::uint32_t>* atomic_n;
  std::uint32_t thread_id;
  std::atomic<bool>* lhs_packed;
  std::atomic<bool>* rhs_packed;
  const Spec& spec;
  TuningResolver* tuning_resolver;
  Allocator* local_allocator;
  Trace* trace;
};

template <typename FixedKernelLayout, typename Scalar, typename PackedScalar>
void CreatePackedMatrix(Tuning tuning, const Matrix<Scalar>& src,
                        Allocator* allocator,
                        Allocator::Handle<PackedScalar>* packed_data_handle,
                        Matrix<PackedScalar>* packed) {
  packed->zero_point = src.zero_point - SymmetricZeroPoint<Scalar>() +
                       SymmetricZeroPoint<PackedScalar>();
  packed->layout = src.layout;
  packed->layout.order = Order::kColMajor;
  packed->layout.rows = round_up_pot(src.layout.rows, FixedKernelLayout::kRows);
  packed->layout.cols = round_up_pot(src.layout.cols, FixedKernelLayout::kCols);
  packed->layout.kernel.order = FixedKernelLayout::kOrder;
  packed->layout.kernel.rows = FixedKernelLayout::kRows;
  packed->layout.kernel.cols = FixedKernelLayout::kCols;
  int innersize = (packed->layout.order == Order::kColMajor)
                      ? packed->layout.rows
                      : packed->layout.cols;
  int outersize = (packed->layout.order == Order::kColMajor)
                      ? packed->layout.cols
                      : packed->layout.rows;
  if (RUY_OPT_SET & RUY_OPT_AVOID_ALIASING) {
    if (tuning == Tuning::kInOrder) {
      packed->layout.stride =
          (innersize * sizeof(Scalar)) % 1024 ? innersize : innersize + 64;
    } else {
      packed->layout.stride =
          (innersize * sizeof(Scalar)) % 4096 ? innersize : innersize + 64;
    }
  } else {
    packed->layout.stride = innersize;
  }
  Reserve(allocator, outersize * packed->layout.stride, packed_data_handle);
}

inline int GetThreadCount(Context* context, int rows, int cols, int depth) {
  // Empirically determined rule for reasonable number of
  // threads to use. This is proportional to the number of arithmetic ops
  // in this Mul (product of the 3 sizes).
  int guess = (std::uint64_t(rows) * cols * depth) >> 13;
  return clamp(guess, 1, context->max_num_threads);
}

template <typename Spec>
LoopStructure GetLoopStructure(int thread_count, int rows, int cols,
                               int depth) {
  if (Spec::kLoopStructure != LoopStructure::kAuto) {
    return Spec::kLoopStructure;
  }
  if (thread_count == 1 &&
      (rows + cols) * depth < kCacheFriendlyLoopThreshold) {
    return LoopStructure::kSimple;
  }
  return LoopStructure::kGeneral;
}

inline Tuning GetTuning(Context* context) {
  context->EnsureNPerThreadStates(1);
  TuningResolver* tuning_resolver =
      &context->per_thread_states[0]->tuning_resolver;
  tuning_resolver->SetTuning(context->explicit_tuning);
  return tuning_resolver->Resolve();
}

// General TrMulImpl definition.  See the reference-code implementation given
// in the partial specialization below for ThePath==kReference.
template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
struct TrMulImpl {
  using AccumScalar = typename Spec::AccumScalar;
  static void Run(const Matrix<LhsScalar>& lhs, const Matrix<RhsScalar>& rhs,
                  const Spec& spec, Context* context, Matrix<DstScalar>* dst) {
    // Fall back, if needed, to Path::kStandardCpp.
    if (ThePath != Path::kStandardCpp) {
      if (!IsLinear(lhs.layout) || !IsLinear(rhs.layout) ||
          !IsLinear(dst->layout) || lhs.layout.order != Order::kColMajor ||
          rhs.layout.order != Order::kColMajor ||
          dst->layout.order != Order::kColMajor) {
        TrMulImpl<Path::kStandardCpp, LhsScalar, RhsScalar, DstScalar,
                  Spec>::Run(lhs, rhs, spec, context, dst);
        return;
      }
    }

    gemmlowp::ScopedProfilingLabel label("TrMulImpl");
    using PackedLhsScalar = PackedType<ThePath, LhsScalar>;
    using PackedRhsScalar = PackedType<ThePath, RhsScalar>;
    using Kernel =
        Kernel<ThePath, PackedLhsScalar, PackedRhsScalar, DstScalar, Spec>;
    using LhsKernelLayout = typename Kernel::LhsLayout;
    using RhsKernelLayout = typename Kernel::RhsLayout;

    const int rows = lhs.layout.cols;
    const int cols = rhs.layout.cols;
    const int depth = lhs.layout.rows;
    const int rows_rounded_up = round_up_pot(rows, LhsKernelLayout::kCols);
    const int cols_rounded_up = round_up_pot(cols, RhsKernelLayout::kCols);

    int thread_count = GetThreadCount(context, rows, cols, depth);
    const auto loop_structure =
        GetLoopStructure<Spec>(thread_count, rows, cols, depth);
    const Tuning tuning = GetTuning(context);
    Allocator* allocator = context->GetMainAllocator();

    // The packed matrices.
    Matrix<PackedLhsScalar> packed_lhs;
    Matrix<PackedRhsScalar> packed_rhs;
    using LhsSumsType = typename Matrix<PackedLhsScalar>::SumsType;
    using RhsSumsType = typename Matrix<PackedRhsScalar>::SumsType;
    const bool lhs_use_packing_sums =
        Pack<PackedRhsScalar>(rhs.zero_point) != 0;
    const bool rhs_use_packing_sums =
        Pack<PackedLhsScalar>(lhs.zero_point) != 0;

    // Allocator handles.
    Allocator::Handle<LhsSumsType> lhs_sums_handle;
    Allocator::Handle<RhsSumsType> rhs_sums_handle;
    Allocator::Handle<PackedLhsScalar> lhs_packed_data_handle;
    Allocator::Handle<PackedRhsScalar> rhs_packed_data_handle;

    // Allocator reservations.
    CreatePackedMatrix<LhsKernelLayout>(tuning, lhs, allocator,
                                        &lhs_packed_data_handle, &packed_lhs);
    CreatePackedMatrix<RhsKernelLayout>(tuning, rhs, allocator,
                                        &rhs_packed_data_handle, &packed_rhs);
    if (lhs_use_packing_sums) {
      Reserve(allocator, rows_rounded_up, &lhs_sums_handle);
    }
    if (rhs_use_packing_sums) {
      Reserve(allocator, cols_rounded_up, &rhs_sums_handle);
    }

    if (loop_structure == LoopStructure::kSimple) {
      gemmlowp::ScopedProfilingLabel label_simple("TrMulImpl, simple loop");

      allocator->Commit();

      // Get pointers from the allocator.
      packed_lhs.data = allocator->GetPointer(lhs_packed_data_handle);
      packed_rhs.data = allocator->GetPointer(rhs_packed_data_handle);
      if (lhs_use_packing_sums) {
        packed_lhs.sums = allocator->GetPointer(lhs_sums_handle);
      }
      if (rhs_use_packing_sums) {
        packed_rhs.sums = allocator->GetPointer(rhs_sums_handle);
      }

      Pack<ThePath, LhsKernelLayout>(tuning, lhs, &packed_lhs, 0,
                                     rows_rounded_up);
      Pack<ThePath, RhsKernelLayout>(tuning, rhs, &packed_rhs, 0,
                                     cols_rounded_up);

      Kernel kernel(tuning);
      RunKernel(kernel, packed_lhs, packed_rhs, spec, 0, 0, rows_rounded_up,
                cols_rounded_up, dst);

      allocator->Decommit();
      return;
    }

    gemmlowp::ScopedProfilingLabel label_general("TrMulImpl, general case");

    auto* trace = NewTraceOrNull(&context->tracing, rows, depth, cols);
    TraceRecordStart(trace);

    // Initialize block map.
    BlockMap block_map;
    MakeBlockMap(rows_rounded_up, cols_rounded_up, depth,
                 LhsKernelLayout::kCols, RhsKernelLayout::kCols,
                 sizeof(LhsScalar), sizeof(RhsScalar), &block_map);
    std::uint16_t num_blocks_of_rows = NumBlocksOfRows(block_map);
    std::uint16_t num_blocks_of_cols = NumBlocksOfCols(block_map);
    std::uint32_t num_blocks = NumBlocks(block_map);
    RUY_DCHECK_EQ(num_blocks, num_blocks_of_rows * num_blocks_of_cols);

    // Initialize per-thread state.
    thread_count = clamp(thread_count, 1, num_blocks);
    context->EnsureNPerThreadStates(thread_count);
    for (auto& per_thread_state : context->per_thread_states) {
      per_thread_state->tuning_resolver.SetTuning(context->explicit_tuning);
    }

    // Allocator handles.
    Allocator::Handle<std::atomic<bool>> lhs_packed_handle;
    Allocator::Handle<std::atomic<bool>> rhs_packed_handle;
    Allocator::Handle<std::atomic<std::uint32_t>> atomic_n_handle;
    using TaskType = TrMulTask<ThePath, LhsScalar, RhsScalar, PackedLhsScalar,
                               PackedRhsScalar, DstScalar, Spec>;
    Allocator::Handle<TaskType> tasks_buffer_handle;
    Allocator::Handle<Task*> tasks_ptrs_handle;

    // Allocator reservations.
    Reserve(allocator, num_blocks_of_rows, &lhs_packed_handle);
    Reserve(allocator, num_blocks_of_cols, &rhs_packed_handle);
    Reserve(allocator, 1, &atomic_n_handle);
    Reserve(allocator, thread_count, &tasks_buffer_handle);
    Reserve(allocator, thread_count, &tasks_ptrs_handle);

    allocator->Commit();

    // Get pointers from the allocator.
    packed_lhs.data = allocator->GetPointer(lhs_packed_data_handle);
    packed_rhs.data = allocator->GetPointer(rhs_packed_data_handle);
    if (lhs_use_packing_sums) {
      packed_lhs.sums = allocator->GetPointer(lhs_sums_handle);
    }
    if (rhs_use_packing_sums) {
      packed_rhs.sums = allocator->GetPointer(rhs_sums_handle);
    }
    std::atomic<bool>* lhs_packed = allocator->GetPointer(lhs_packed_handle);
    std::atomic<bool>* rhs_packed = allocator->GetPointer(rhs_packed_handle);
    std::atomic<std::uint32_t>* atomic_n =
        allocator->GetPointer(atomic_n_handle);

    // Initialize allocated data.
    for (int i = 0; i < num_blocks_of_rows; i++) {
      lhs_packed[i].store(false, std::memory_order_release);
    }
    for (int i = 0; i < num_blocks_of_cols; i++) {
      rhs_packed[i].store(false, std::memory_order_release);
    }
    atomic_n->store(thread_count);

    TaskType* tasks = allocator->GetPointer(tasks_buffer_handle);
    Task** tasks_ptrs = allocator->GetPointer(tasks_ptrs_handle);

    for (int i = 0; i < thread_count; i++) {
      tasks_ptrs[i] = static_cast<Task*>(tasks + i);
      new (tasks_ptrs[i])
          TaskType(lhs, rhs, &packed_lhs, &packed_rhs, dst, block_map, atomic_n,
                   i, lhs_packed, rhs_packed, spec,
                   &context->per_thread_states[i]->tuning_resolver,
                   &context->per_thread_states[i]->allocator, trace);
    }

    // Do the computation.
    TraceRecordExecute(trace);
    TraceStartRecordingBlockAndThreadFields(block_map, thread_count, trace);

    context->workers_pool.Execute(thread_count, tasks_ptrs);

    // Finish up.
    for (int i = 0; i < thread_count; i++) {
      tasks[i].~TaskType();
    }

    TraceRecordEnd(trace);

    allocator->Decommit();
  }
};

// Reference code for TrMul, doing a transpose-multiply: compute
//   Destination = Transpose(LHS) * RHS
template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename Spec>
struct TrMulImpl<Path::kReference, LhsScalar, RhsScalar, DstScalar, Spec> {
  static void Run(const Matrix<LhsScalar>& lhs, const Matrix<RhsScalar>& rhs,
                  const Spec& spec, Context*, Matrix<DstScalar>* dst) {
    gemmlowp::ScopedProfilingLabel label("TrMulImpl Reference");
    for (int i = 0; i < lhs.layout.cols; i++) {
      for (int j = 0; j < rhs.layout.cols; j++) {
        using AccumScalar = typename Spec::AccumScalar;
        AccumScalar accum = 0;
        for (int k = 0; k < lhs.layout.rows; k++) {
          AccumScalar lhs_val = Element(lhs, k, i);
          AccumScalar rhs_val = Element(rhs, k, j);
          accum += (lhs_val - lhs.zero_point) * (rhs_val - rhs.zero_point);
        }
        if (spec.bias) {
          accum += spec.bias[i];
        }
        ApplyMultiplier(spec, i, &accum);
        accum += dst->zero_point;
        accum = std::min<AccumScalar>(accum, spec.clamp_max);
        accum = std::max<AccumScalar>(accum, spec.clamp_min);
        *ElementPtr(dst, i, j) = static_cast<DstScalar>(accum);
      }
    }
  }
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_IMPL_H_

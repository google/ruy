#include "trace.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "block_map.h"
#include "check_macros.h"
#include "time.h"

namespace ruy {

#ifdef RUY_TRACE

struct BlockTraceEntry {
  std::uint32_t thread_id = 0;
  std::int64_t time_reserved = 0;
  std::int64_t time_computed_coords = 0;
  std::int64_t time_packed_lhs = 0;
  std::int64_t time_packed_rhs = 0;
  std::int64_t time_finished = 0;
};

struct ThreadTraceEntry {
  std::int64_t time_start = 0;
  std::int64_t time_loop_start = 0;
  std::int64_t time_end = 0;
};

struct Trace {
  enum class LifeStage {
    kInitial,
    kRecordingRootFields,
    kRecordingBlockAndThreadFields,
    kComplete
  };
  void StartRecordingBlockAndThreadFields(const BlockMap& block_map_,
                                          int thread_count_) {
    RUY_DCHECK(life_stage == LifeStage::kRecordingRootFields);
    block_map = block_map_;
    thread_count = thread_count_;
    int num_blocks = NumBlocks(block_map);
    if (num_blocks > block_entries.size()) {
      block_entries.resize(NumBlocks(block_map));
    }
    if (thread_count > thread_entries.size()) {
      thread_entries.resize(thread_count);
    }
    memset(block_entries.data(), 0,
           sizeof(block_entries[0]) * block_entries.size());
    memset(thread_entries.data(), 0,
           sizeof(thread_entries[0]) * thread_entries.size());
    life_stage = LifeStage::kRecordingBlockAndThreadFields;
  }
  BlockMap block_map;
  int thread_count = 0;
  std::vector<BlockTraceEntry> block_entries;
  std::vector<ThreadTraceEntry> thread_entries;
  std::int64_t time_start = 0;
  std::int64_t time_execute = 0;
  std::int64_t time_end = 0;
  double frequency = 0;
  LifeStage life_stage = LifeStage::kInitial;
};

struct ProcessedTrace {
  enum class Event : std::uint8_t {
    kNone,
    kThreadStart,
    kThreadLoopStart,
    kThreadEnd,
    kBlockReserved,
    kBlockComputedCoords,
    kBlockPackedLhs,
    kBlockPackedRhs,
    kBlockFinished
  };
  struct Entry {
    Event event = Event::kNone;
    std::uint32_t thread_id = 0;
    std::uint32_t block_id = 0;
    std::int64_t time = 0;
  };

  BlockMap block_map;
  int thread_count = 0;
  std::int64_t time_start = 0;
  std::int64_t time_execute = 0;
  std::int64_t time_end = 0;
  double frequency = 0;
  std::vector<Entry> entries;
  void Add(Event event, std::uint32_t thread_id, std::uint32_t block_id,
           std::int64_t time) {
    if (!time) {
      return;
    }
    Entry entry;
    entry.event = event;
    entry.thread_id = thread_id;
    entry.block_id = block_id;
    entry.time = time;
    entries.push_back(entry);
  }
  void Process(const Trace& trace) {
    thread_count = trace.thread_count;
    block_map = trace.block_map;
    frequency = trace.frequency;
    time_start = trace.time_start;
    time_execute = trace.time_execute;
    time_end = trace.time_end;
    entries.clear();
    for (int i = 0; i < trace.thread_count; i++) {
      const auto& entry = trace.thread_entries[i];
      Add(Event::kThreadStart, i, 0, entry.time_start);
      Add(Event::kThreadLoopStart, i, 0, entry.time_loop_start);
      Add(Event::kThreadEnd, i, 0, entry.time_end);
    }
    std::uint32_t num_blocks = NumBlocks(block_map);
    for (int i = 0; i < num_blocks; i++) {
      const auto& entry = trace.block_entries[i];
      Add(Event::kBlockReserved, entry.thread_id, i, entry.time_reserved);
      Add(Event::kBlockComputedCoords, entry.thread_id, i,
          entry.time_computed_coords);
      Add(Event::kBlockPackedLhs, entry.thread_id, i, entry.time_packed_lhs);
      Add(Event::kBlockPackedRhs, entry.thread_id, i, entry.time_packed_rhs);
      Add(Event::kBlockFinished, entry.thread_id, i, entry.time_finished);
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) -> bool {
                return a.time < b.time ||
                       (a.time == b.time &&
                        static_cast<int>(a.event) < static_cast<int>(b.event));
              });
  }
  void Dump() {
    const char* trace_filename = getenv("RUY_TRACE_FILE");
    FILE* trace_file = trace_filename ? fopen(trace_filename, "w") : stderr;
    if (!trace_file) {
      fprintf(stderr, "Failed to open %s for write, errno=%d\n", trace_filename,
              errno);
      RUY_CHECK(false);
    }
    fprintf(trace_file, "thread_count:%d\n", thread_count);
    fprintf(trace_file, "num_blocks:%d\n", NumBlocks(block_map));
    fprintf(trace_file, "rows:%d\n", block_map.rows);
    fprintf(trace_file, "cols:%d\n", block_map.cols);
    fprintf(trace_file, "frequency:%g\n", frequency);
    fprintf(trace_file, "Execute: %ld\n", time_execute - time_start);
    for (const Entry& entry : entries) {
      std::int64_t time = entry.time - time_start;
      switch (entry.event) {
        case Event::kThreadStart:
          fprintf(trace_file, "ThreadStart: %ld, %d\n", time, entry.thread_id);
          break;
        case Event::kThreadLoopStart:
          fprintf(trace_file, "ThreadLoopStart: %ld, %d\n", time,
                  entry.thread_id);
          break;
        case Event::kThreadEnd:
          fprintf(trace_file, "ThreadEnd: %ld, %d\n", time, entry.thread_id);
          break;
        case Event::kBlockReserved: {
          std::uint16_t block_r, block_c;
          int start_r, start_c, end_r, end_c;
          GetBlockByIndex(block_map, entry.block_id, &block_r, &block_c);
          GetBlockMatrixCoords(block_map, block_r, block_c, &start_r, &start_c,
                               &end_r, &end_c);
          fprintf(trace_file, "BlockReserved: %ld, %d, %d, %d, %d, %d, %d\n",
                  time, entry.thread_id, entry.block_id, start_r, start_c,
                  end_r, end_c);
          break;
        }
        case Event::kBlockComputedCoords:
          fprintf(trace_file, "BlockComputedCoords: %ld, %d, %d\n", time,
                  entry.thread_id, entry.block_id);
          break;
        case Event::kBlockPackedLhs:
          fprintf(trace_file, "BlockPackedLhs: %ld, %d, %d\n", time,
                  entry.thread_id, entry.block_id);
          break;
        case Event::kBlockPackedRhs:
          fprintf(trace_file, "BlockPackedRhs: %ld, %d, %d\n", time,
                  entry.thread_id, entry.block_id);
          break;
        case Event::kBlockFinished:
          fprintf(trace_file, "BlockFinished: %ld, %d, %d\n", time,
                  entry.thread_id, entry.block_id);
          break;
        default:
          RUY_CHECK(false);
      }
    }
    fprintf(trace_file, "End: %ld\n", time_end - time_start);
    if (trace_filename) {
      fclose(trace_file);
    }
  }
};

void DumpTrace(const Trace& trace) {
  ProcessedTrace processed_trace;
  processed_trace.Process(trace);
  processed_trace.Dump();
}

Trace* GetTraceOrNull(TracingContext* tracing, int rows, int depth, int cols) {
  if (!tracing->initialized) {
    tracing->initialized = true;
    tracing->enabled = getenv("RUY_TRACE");
    if (!tracing->enabled) {
      return nullptr;
    }
    if (getenv("RUY_TRACE_FILTER_ROWS")) {
      tracing->filter_shape_rows = std::stoi(getenv("RUY_TRACE_FILTER_ROWS"));
    }
    if (getenv("RUY_TRACE_FILTER_DEPTH")) {
      tracing->filter_shape_depth = std::stoi(getenv("RUY_TRACE_FILTER_DEPTH"));
    }
    if (getenv("RUY_TRACE_FILTER_COLS")) {
      tracing->filter_shape_cols = std::stoi(getenv("RUY_TRACE_FILTER_COLS"));
    }
  }
  if (!tracing->enabled) {
    return nullptr;
  }
  if (tracing->filter_shape_rows && rows != tracing->filter_shape_rows) {
    return nullptr;
  }
  if (tracing->filter_shape_depth && depth != tracing->filter_shape_depth) {
    return nullptr;
  }
  if (tracing->filter_shape_cols && cols != tracing->filter_shape_cols) {
    return nullptr;
  }
  if (!tracing->trace) {
    tracing->trace = new Trace;
  }
  return tracing->trace;
}

TracingContext::~TracingContext() {
  if (trace) {
    DumpTrace(*trace);
  }
  delete trace;
}

void TraceRecordThreadStart(std::uint32_t thread_id, Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage ==
               Trace::LifeStage::kRecordingBlockAndThreadFields);
    trace->block_entries[thread_id].thread_id = thread_id;
    trace->block_entries[thread_id].time_reserved =
        trace->thread_entries[thread_id].time_start = TimeNowBarrier();
  }
}

void TraceRecordThreadLoopStart(std::uint32_t thread_id, Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage ==
               Trace::LifeStage::kRecordingBlockAndThreadFields);
    trace->thread_entries[thread_id].time_loop_start = TimeNowBarrier();
  }
}

void TraceRecordBlockReserved(std::uint32_t thread_id, std::uint32_t block_id,
                              Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage ==
               Trace::LifeStage::kRecordingBlockAndThreadFields);
    // This is typically called on the next block id just obtained by atomic
    // increment; this may be out of range.
    if (block_id < trace->block_entries.size()) {
      trace->block_entries[block_id].thread_id = thread_id;
      trace->block_entries[block_id].time_reserved = TimeNowBarrier();
    }
  }
}

void TraceRecordBlockCoordsComputed(std::uint32_t block_id, Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage ==
               Trace::LifeStage::kRecordingBlockAndThreadFields);
    trace->block_entries[block_id].time_computed_coords = TimeNowBarrier();
  }
}

void TraceRecordBlockPackedLhs(std::uint32_t block_id, Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage ==
               Trace::LifeStage::kRecordingBlockAndThreadFields);
    trace->block_entries[block_id].time_packed_lhs = TimeNowBarrier();
  }
}

void TraceRecordBlockPackedRhs(std::uint32_t block_id, Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage ==
               Trace::LifeStage::kRecordingBlockAndThreadFields);
    trace->block_entries[block_id].time_packed_rhs = TimeNowBarrier();
  }
}

void TraceRecordBlockFinished(std::uint32_t block_id, Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage ==
               Trace::LifeStage::kRecordingBlockAndThreadFields);
    trace->block_entries[block_id].time_finished = TimeNowBarrier();
  }
}

void TraceRecordThreadEnd(std::uint32_t thread_id, Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage ==
               Trace::LifeStage::kRecordingBlockAndThreadFields);
    trace->thread_entries[thread_id].time_end = TimeNowBarrier();
  }
}

void TraceRecordStart(Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage == Trace::LifeStage::kInitial ||
               trace->life_stage == Trace::LifeStage::kComplete);
    trace->time_start = 0;
    trace->time_execute = 0;
    trace->time_end = 0;
    trace->frequency = 0;
    trace->time_start = TimeNowBarrier();
    trace->life_stage = Trace::LifeStage::kRecordingRootFields;
  }
}

void TraceRecordExecute(Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage == Trace::LifeStage::kRecordingRootFields);
    trace->time_execute = TimeNowBarrier();
  }
}

void TraceRecordEnd(Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage ==
               Trace::LifeStage::kRecordingBlockAndThreadFields);
    trace->time_end = TimeNowBarrier();
    trace->frequency = TimeFrequency();
    trace->life_stage = Trace::LifeStage::kComplete;
  }
}

void TraceStartRecordingBlockAndThreadFields(const BlockMap& block_map,
                                             int thread_count, Trace* trace) {
  if (trace) {
    RUY_DCHECK(trace->life_stage == Trace::LifeStage::kRecordingRootFields);
    trace->StartRecordingBlockAndThreadFields(block_map, thread_count);
    trace->life_stage = Trace::LifeStage::kRecordingBlockAndThreadFields;
  }
}

#endif

}  // namespace ruy

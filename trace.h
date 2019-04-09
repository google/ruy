#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRACE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRACE_H_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "block_map.h"

namespace ruy {

struct Trace;

#ifdef RUY_TRACE

struct TracingContext {
  bool initialized = false;
  bool enabled = false;
  int filter_shape_rows = 0;
  int filter_shape_cols = 0;
  int filter_shape_depth = 0;
  Trace* trace = nullptr;
  ~TracingContext();
};

void DumpTrace(const Trace& trace);

Trace* GetTraceOrNull(TracingContext* context, int rows, int depth, int cols);
void TraceRecordThreadStart(std::uint32_t thread_id, Trace* trace);
void TraceRecordThreadLoopStart(std::uint32_t thread_id, Trace* trace);
void TraceRecordBlockReserved(std::uint32_t thread_id, std::uint32_t block_id,
                              Trace* trace);
void TraceRecordBlockCoordsComputed(std::uint32_t block_id, Trace* trace);
void TraceRecordBlockPackedLhs(std::uint32_t block_id, Trace* trace);
void TraceRecordBlockPackedRhs(std::uint32_t block_id, Trace* trace);
void TraceRecordBlockFinished(std::uint32_t block_id, Trace* trace);
void TraceRecordThreadEnd(std::uint32_t thread_id, Trace* trace);
void TraceRecordStart(Trace* trace);
void TraceRecordExecute(Trace* trace);
void TraceRecordEnd(Trace* trace);
void TraceStartRecordingBlockAndThreadFields(const BlockMap& block_map,
                                             int thread_count, Trace* trace);

#else

struct TracingContext {};

inline Trace* GetTraceOrNull(TracingContext*, int, int, int) { return nullptr; }

inline void TraceRecordThreadStart(std::uint32_t, Trace*) {}
inline void TraceRecordThreadLoopStart(std::uint32_t, Trace*) {}
inline void TraceRecordBlockReserved(std::uint32_t, std::uint32_t, Trace*) {}
inline void TraceRecordBlockCoordsComputed(std::uint32_t, Trace*) {}
inline void TraceRecordBlockPackedLhs(std::uint32_t, Trace*) {}
inline void TraceRecordBlockPackedRhs(std::uint32_t, Trace*) {}
inline void TraceRecordBlockFinished(std::uint32_t, Trace*) {}
inline void TraceRecordThreadEnd(std::uint32_t, Trace*) {}
inline void TraceRecordStart(Trace*) {}
inline void TraceRecordExecute(Trace*) {}
inline void TraceRecordEnd(Trace*) {}
inline void TraceStartRecordingBlockAndThreadFields(const BlockMap&, int,
                                                    Trace*) {}

#endif

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRACE_H_
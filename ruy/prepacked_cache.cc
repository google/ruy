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

#include "ruy/prepacked_cache.h"

#include <utility>

#include "ruy/mat.h"
#include "ruy/profiler/instrumentation.h"

namespace ruy {

using CacheIterator = PrepackedCache::CacheIterator;

// Looks for an entry with `key`. If found, update its time stamp.
CacheIterator PrepackedCache::FindAndUpdate(const CacheKey &key) {
  auto itr = cache_.find(key);
  // If found, update with new access time for this entry.
  if (itr != cache_.end()) {
    const TimePoint time = CacheNow();
    itr->second.second = time;
  }
#ifdef _MSC_VER
  // std::move() is required in MSVC when NDEBUG is not set
  return std::move(itr);  // NOLINT
#else
  return itr;
#endif
}

void PrepackedCache::Insert(const CacheKey &key, const PEMat &matrix) {
  // Calculate size of this new item.
  const int size_bytes = DataBytes(matrix) + SumsBytes(matrix);

  // While we are above the threshold of ejection, eject the LRU entry.
  while (!cache_.empty() &&
         ((TotalSize() + size_bytes) > ejection_threshold_)) {
    EjectOne();
  }
  DoInsert(key, matrix);
  cache_size_ += size_bytes;
}

void PrepackedCache::EjectOne() {
  TimePoint oldest_time = CacheNow();
  auto oldest = cache_.begin();
  {
    profiler::ScopeLabel label("PrepackedCacheEjection");
    for (auto itr = cache_.begin(); itr != cache_.end(); ++itr) {
      if (itr->second.second < oldest_time) {
        oldest_time = itr->second.second;
        oldest = itr;
      }
    }
  }
  PEMat &matrix = oldest->second.first;
  cache_size_ -= DataBytes(matrix) + SumsBytes(matrix);

  allocator_.Free(matrix.data);
  allocator_.Free(matrix.sums);
  cache_.erase(oldest);
}

void PrepackedCache::AllocatePrepackedMatrix(PEMat *matrix) {
  matrix->data = allocator_.Alloc(DataBytes(*matrix));
  matrix->sums = allocator_.Alloc(SumsBytes(*matrix));
}

void PrepackedCache::DoInsert(const CacheKey &key, const PEMat &matrix) {
  const TimePoint t = CacheNow();
  const MatrixWithTimeStamp mts({matrix, t});
  cache_.insert({key, mts});
}

}  // namespace ruy

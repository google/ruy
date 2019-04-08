// This file is a fork of gemmlowp's multi_thread_gemm.h, under Apache 2.0
// license.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_THREAD_POOL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_THREAD_POOL_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "blocking_counter.h"

namespace ruy {

// A workload for a thread.
struct Task {
  virtual ~Task() {}
  virtual void Run() = 0;
};

class Thread;

// A very simple pool of threads, that only allows the very
// specific parallelization pattern that we use here:
// a fixed number of threads can be given work, and one then
// waits for all of them to finish.
//
// See MultiThreadGemmContextBase for how other ThreadPool implementations can
// be used.
class ThreadPool {
 public:
  ThreadPool() {}

  ~ThreadPool();

  void Execute(int thread_count, Task** tasks_ptrs);

 private:
  // Ensures that the pool has at least the given count of threads.
  // If any new thread has to be created, this function waits for it to
  // be ready.
  void CreateThreads(std::size_t threads_count);

  // copy construction disallowed
  ThreadPool(const ThreadPool&) = delete;

  // The threads in this pool. They are owned by the pool:
  // the pool creates threads and destroys them in its destructor.
  std::vector<Thread*> threads_;

  // The BlockingCounter used to wait for the threads.
  BlockingCounter counter_to_decrement_when_ready_;
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_THREAD_POOL_H_

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

/* Temporary dotprod-detection until we can rely on proper feature-detection
such as getauxval on Linux (requires a newer Linux kernel than we can
currently rely on on Android).

There are two main ways that this could be implemented: using a signal
handler or a fork. The current implementation uses a signal handler.
This is because on current Android, an uncaught signal gives a latency
of over 100 ms. In order for the fork approach to be worthwhile, it would
have to save us the hassle of handling signals, and such an approach thus
has an unavoidable 100ms latency. By contrast, the present signal-handling
approach has low latency.

Downsides of the current signal-handling approach include:
 1. Setting and restoring signal handlers is not thread-safe: we can't
    prevent another thread from interfering with us. We at least prevent
    other threads from calling our present code concurrently by using a lock,
    but we can't do anything about other threads using their own code to
    set signal handlers.
 2. Signal handlers are not entirely portable, e.g. b/132973173 showed that
    on Apple platform the EXC_BAD_INSTRUCTION signal is not always caught
    by a SIGILL handler (difference between Release and Debug builds).
 3. The signal handler approach looks confusing in a debugger (has to
    tell the debugger to 'continue' past the signal every time). Fix:
    ```
    (gdb) handle SIGILL nostop noprint pass
    ```

Here is what the nicer fork-based alternative would look like.
Its only downside, as discussed above, is high latency, 100 ms on Android.

```
bool TryAsmSnippet(bool (*asm_snippet)()) {
  int child_pid = fork();
  if (child_pid == -1) {
    // Fork failed.
    return false;
  }
  if (child_pid == 0) {
    // Child process code path. Pass the raw boolean return value of
    // asm_snippet as exit code (unconventional: 1 means true == success).
    _exit(asm_snippet());
  }

  int child_status;
  waitpid(child_pid, &child_status, 0);
  if (WIFSIGNALED(child_status)) {
    // Child process terminated by signal, meaning the instruction was
    // not supported.
    return false;
  }
  // Return the exit code of the child, which per child code above was
  // the return value of asm_snippet().
  return WEXITSTATUS(child_status);
}
```
*/

#include "ruy/detect_arm.h"

#include "ruy/platform.h"

#if RUY_PLATFORM(NEON_DETECT_DOTPROD)

#include <setjmp.h>
#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <tuple>

#ifdef __linux__
#include <sys/auxv.h>
#endif

#endif

namespace ruy {

#if RUY_PLATFORM(NEON_DETECT_DOTPROD)

namespace {

// long-jump buffer used to continue execution after a caught SIGILL.
sigjmp_buf global_sigjmp_buf;

// Signal handler. Long-jumps to just before
// we ran the snippet that we know is the only thing that could have generated
// the SIGILL.
void SignalHandler(int) { siglongjmp(global_sigjmp_buf, 1); }

// RAII helper for calling sigprocmask to unblock all signals temporarily.
class ScopeUnblockSignals final {
 public:
  ScopeUnblockSignals() {
    sigset_t procmask;
    sigemptyset(&procmask);
    success_ = !sigprocmask(SIG_SETMASK, &procmask, &old_procmask_);
  }
  ~ScopeUnblockSignals() {
    if (success_) {
      sigprocmask(SIG_SETMASK, &old_procmask_, nullptr);
    }
  }
  bool success() const { return success_; }

 private:
  sigset_t old_procmask_;
  bool success_ = false;
};

// RAII helper to install and uninstall a signal handler.
class ScopeSigaction final {
 public:
  ScopeSigaction(int signal_number, void (*handler_function)(int))
      : signal_number_(signal_number) {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    sigemptyset(&action.sa_mask);
    action.sa_handler = handler_function;
    success_ = !sigaction(signal_number_, &action, &old_action_);
  }
  ~ScopeSigaction() {
    if (success_) {
      sigaction(signal_number_, &old_action_, nullptr);
    }
  }
  bool success() const { return success_; }

 private:
  const int signal_number_;
  struct sigaction old_action_;
  bool success_ = false;
};

// Try an asm snippet. Returns true if it passed i.e. ran without generating
// an illegal-instruction signal and returned true. Returns false otherwise.
bool TryAsmSnippet(bool (*asm_snippet)()) {
  // This function installs and restores signal handlers and the signal-blocking
  // mask. We can't prevent another thread from interfering, but we can at least
  // put a big lock here so that it works if, for whatever reason, another
  // thread calls this function concurrently.
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  ScopeUnblockSignals unblock_signals;
  if (!unblock_signals.success()) {
    return false;
  }
  ScopeSigaction handle_sigill(SIGILL, SignalHandler);
  if (!handle_sigill.success()) {
    return false;
  }
#ifdef EXC_BAD_INSTRUCTION
  // On Apple platforms, we also need to handle this signal, in addition to
  // handling SIGILL.
  ScopeSigaction handle_exc_bad_instruction(EXC_BAD_INSTRUCTION, SignalHandler);
  if (!handle_exc_bad_instruction.success()) {
    return false;
  }
#endif

  // Set the long jump buffer to this point in the code. This normally returns
  // 0 so we don't take this branch...
  if (sigsetjmp(global_sigjmp_buf, false)) {
    // ... except in the fake return from sigsetjmp that is produced when
    // the long-jump back to here actually happened, that is, in the signal
    // handler. In this case, we know that the asm_snippet triggered an illegal
    // instruction signal, so we return false.
    return false;
  }

  return asm_snippet();
}

bool DotprodAsmSnippet() {
  // maratek@ mentioned that for some other ISA extensions (fp16)
  // there have been implementations that failed to generate SIGILL even
  // though they did not correctly implement the instruction. Just in case
  // a similar situation might exist here, we do a simple correctness test.
  int result = 0;
  asm volatile(
      "mov w0, #100\n"
      "dup v0.16b, w0\n"
      "dup v1.4s, w0\n"
      ".word 0x6e809401  // udot v1.4s, v0.16b, v0.16b\n"
      "mov %w[result], v1.s[0]\n"
      : [result] "=r"(result)
      :
      : "x0", "v0", "v1");
  // Expecting 100 (input accumulator value) + 100 * 100 + ... (repeat 4 times)
  return result == 40100;
}

bool DetectDotprodBySignalMethod() { return TryAsmSnippet(DotprodAsmSnippet); }

#ifdef __linux__
bool DetectDotprodByLinuxAuxvMethod() {
  // This is the value of HWCAP_ASIMDDP in sufficiently recent Linux headers,
  // however we need to support building against older headers for the time
  // being.
  const int kLocalHwcapAsimddp = 1 << 20;
  return getauxval(AT_HWCAP) & kLocalHwcapAsimddp;
}
#endif

}  // namespace

bool DetectDotprod() {
#ifdef __linux__
  // We always try the auxv method and don't try to check the linux version
  // before. It's only in the mainline linux tree from 4.14.151, but it's been
  // backported to earlier linux versions in Android vendor device trees.
  // The cost of just trying this is near zero, and the benefit is large
  // as the signal method has higher latency and a substantial crash potential.
  if (DetectDotprodByLinuxAuxvMethod()) {
    return true;
  }
#endif

  return DetectDotprodBySignalMethod();
}

#else   // not RUY_PLATFORM(NEON_DETECT_DOTPROD)
bool DetectDotprod() { return false; }
#endif  // RUY_PLATFORM(NEON_DETECT_DOTPROD)

}  // namespace ruy

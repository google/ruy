// Temporary dotprod-detection code until we can rely on getauxval.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_DETECT_DOTPROD_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_DETECT_DOTPROD_H_

namespace ruy {

// On A64, returns true if the dotprod extension is present.
// On other architectures, returns false unconditionally.
bool DetectDotprod();

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_DETECT_DOTPROD_H_

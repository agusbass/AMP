// profiler.hpp — vendor-agnostic GPU profiler annotation macros
// Zero overhead when AMP_ENABLE_PROFILER is NOT defined (all macros → no-op).
// To enable, build with -DAMP_ENABLE_PROFILER:
//   CUDA:  link -lnvToolsExt   (or use NVTX3 header-only in CUDA 12+)
//   HIP:   link -lroctx64      (ROCm 6+)
//   SYCL:  link -littnotify    (Intel oneAPI 2024+)
#pragma once

#if defined(AMP_BACKEND_CUDA) && defined(AMP_ENABLE_PROFILER)
  #include <nvToolsExt.h>
  #define AMP_RANGE_PUSH(name)  nvtxRangePushA(name)
  #define AMP_RANGE_POP()       nvtxRangePop()
  #define AMP_MARK(name)        nvtxMarkA(name)

#elif defined(AMP_BACKEND_HIP) && defined(AMP_ENABLE_PROFILER)
  #include <roctx.h>
  #define AMP_RANGE_PUSH(name)  roctxRangePushA(name)
  #define AMP_RANGE_POP()       roctxRangePop()
  #define AMP_MARK(name)        roctxMarkA(name)

#elif defined(AMP_BACKEND_SYCL) && defined(AMP_ENABLE_PROFILER)
  #include <ittnotify.h>
  namespace AMP { namespace detail {
    inline __itt_domain* itt_domain() noexcept {
      static __itt_domain* d = __itt_domain_create("AMP");
      return d;
    }
  }}
  #define AMP_RANGE_PUSH(name) \
      __itt_task_begin(AMP::detail::itt_domain(), __itt_null, __itt_null, \
                       __itt_string_handle_create(name))
  #define AMP_RANGE_POP()      __itt_task_end(AMP::detail::itt_domain())
  #define AMP_MARK(name)       ((void)0)

#else
  #define AMP_RANGE_PUSH(name) ((void)0)
  #define AMP_RANGE_POP()      ((void)0)
  #define AMP_MARK(name)       ((void)0)
#endif

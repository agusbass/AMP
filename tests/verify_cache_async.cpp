// verify_cache_async.cpp - stress-test flush() durability guarantee.
// Before the fix, flush() could return while the last queued write was
// still in-flight (notify happened before disk_.put() completed).
//
// This target is meant to be backend-agnostic (no GPU touched), but CMake's
// add_compile_definitions(AMP_BACKEND_CUDA/HIP/...) is a directory-scoped
// property that applies to every target in this CMakeLists.txt regardless
// of where add_executable() is called relative to it. #undef here forces
// portable.hpp's plain CPU stub path no matter what other targets need.
#ifdef AMP_BACKEND_CUDA
#undef AMP_BACKEND_CUDA
#endif
#ifdef AMP_BACKEND_HIP
#undef AMP_BACKEND_HIP
#endif
#ifdef AMP_BACKEND_SYCL
#undef AMP_BACKEND_SYCL
#endif

#include "cache_async.hpp"
#include <cstdio>
#include <filesystem>

int main() {
    std::filesystem::remove_all("/tmp/amp_cache_test");
    int failures = 0;

    for (int iter = 0; iter < 200; ++iter) {
        AMP::CacheStoreAsync store("/tmp/amp_cache_test");
        const int N = 50;
        for (int i = 0; i < N; ++i) {
            std::vector<char> bin = {char(i), char(i + 1)};
            store.put("key" + std::to_string(i), bin);
        }
        store.flush();
        uint64_t queued = store.stats().writes_queued.load();
        uint64_t persisted = store.stats().writes_persisted.load();
        if (queued != persisted) {
            printf("FAIL iter %d: queued=%lu persisted=%lu (flush() returned before all writes persisted)\n",
                   iter, (unsigned long)queued, (unsigned long)persisted);
            failures++;
        }
    }

    printf("%s (200 iterations x 50 writes, flush() durability check)\n",
           failures == 0 ? "PASS: writes_persisted always matched writes_queued after flush()"
                         : "FAIL: see above");
    return failures == 0 ? 0 : 1;
}

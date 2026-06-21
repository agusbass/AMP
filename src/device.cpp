// device.cpp - runtime device detection per backend
#include "portable.hpp"
#include <thread>
#include <sstream>
#include <iomanip>

DeviceInfo detect_device(int device_id) {
    DeviceInfo info{};
    info.sm_major = 0; info.sm_minor = 0;

#if defined(AMP_BACKEND_CUDA)
    cudaDeviceProp p;
    AMP_CHECK(cudaGetDeviceProperties(&p, device_id));
    info.vendor      = "nvidia";
    info.name        = p.name;
    info.simd_width  = p.warpSize;
    info.compute_units = p.multiProcessorCount;
    info.global_mem  = p.totalGlobalMem;
    info.sm_major    = p.major;
    info.sm_minor    = p.minor;
    // tensor_core: Volta SM7.0+; BF16: SM8.0+; FP8: SM8.9+ (Ada/Hopper)
    if      ((p.major > 8) || (p.major == 8 && p.minor >= 9))
                             info.matrix_unit = "tensor_core_fp8";  // RTX4090, H100
    else if (p.major >= 8)   info.matrix_unit = "tensor_core_bf16"; // A100
    else if (p.major >= 7)   info.matrix_unit = "tensor_core";      // V100
    else                     info.matrix_unit = "none";

#elif defined(AMP_BACKEND_HIP)
    hipDeviceProp_t p;
    AMP_CHECK(hipGetDeviceProperties(&p, device_id));
    info.vendor      = "amd";
    info.name        = p.name;
    info.simd_width  = p.warpSize;   // 64 CDNA, 32 RDNA
    info.compute_units = p.multiProcessorCount;
    info.global_mem  = p.totalGlobalMem;
    std::string arch = p.gcnArchName;
    // gfx940/941/942 = MI300X (MFMA FP8 + BF16); gfx90a = MI200 (BF16); gfx908 = MI100
    if (arch.find("gfx940") != std::string::npos ||
        arch.find("gfx941") != std::string::npos ||
        arch.find("gfx942") != std::string::npos) {
        info.matrix_unit = "mfma_fp8";    // MI300X — FP8 + BF16 MFMA
    } else if (arch.find("gfx90a") != std::string::npos ||
               arch.find("gfx908") != std::string::npos) {
        info.matrix_unit = "mfma_bf16";   // MI200 — BF16 MFMA
    } else {
        info.matrix_unit = "none";
    }

#elif defined(AMP_BACKEND_SYCL)
    try {
        auto dev = AMP::detail::default_q().get_device();
        info.vendor = "intel";
        info.name   = dev.get_info<sycl::info::device::name>();
        info.simd_width  = dev.get_info<sycl::info::device::sub_group_sizes>().back();
        info.compute_units = (int)dev.get_info<sycl::info::device::max_compute_units>();
        info.global_mem  = dev.get_info<sycl::info::device::global_mem_size>();
        // XMX = Intel Matrix Extensions (Xe-HPC/Gaudi3, Arc)
        bool has_xmx = dev.has(sycl::aspect::ext_intel_matrix);
        info.matrix_unit = has_xmx ? "xmx" : "none";
    } catch (...) {
        info.vendor = "intel"; info.name = "unknown"; info.matrix_unit = "none";
    }

#else // CPU
    info.vendor      = "cpu";
    info.name        = "cpu";
    info.simd_width  = 1;
    info.compute_units = std::thread::hardware_concurrency();
    info.global_mem  = 0;
    info.matrix_unit = "none";
#endif
    return info;
}

DeviceCaps detect_caps(int device_id) {
    DeviceCaps caps;
#if defined(AMP_BACKEND_CUDA)
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, device_id) != cudaSuccess) return caps;
    caps.has_fp16 = (p.major >= 6);                            // Pascal+
    caps.has_bf16 = (p.major >= 8);                            // Ampere+
    caps.has_int8 = (p.major >= 6 && p.minor >= 1) || (p.major >= 7); // DP4A
    caps.has_fp8  = (p.major > 8) || (p.major == 8 && p.minor >= 9);  // SM8.9+
    caps.has_flash_attn = caps.has_bf16 &&
                          p.sharedMemPerMultiprocessor >= 80 * 1024;
    caps.shared_mem_kb  = (int)(p.sharedMemPerMultiprocessor / 1024);
    cudaGetDeviceCount(&caps.n_devices);

#elif defined(AMP_BACKEND_HIP)
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, device_id) != hipSuccess) return caps;
    std::string arch = p.gcnArchName;
    caps.has_fp16 = true;  // all modern AMD GPUs
    caps.has_bf16 = arch.find("gfx90a")  != std::string::npos ||
                   arch.find("gfx940")  != std::string::npos ||
                   arch.find("gfx941")  != std::string::npos ||
                   arch.find("gfx942")  != std::string::npos;
    caps.has_fp8  = arch.find("gfx940") != std::string::npos ||
                   arch.find("gfx941") != std::string::npos ||
                   arch.find("gfx942") != std::string::npos;
    caps.has_int8 = caps.has_bf16;
    caps.has_flash_attn = caps.has_bf16;
    caps.shared_mem_kb  = (int)(p.sharedMemPerBlock / 1024);
    hipGetDeviceCount(&caps.n_devices);

#elif defined(AMP_BACKEND_SYCL)
    try {
        auto dev = AMP::detail::default_q().get_device();
        caps.has_bf16  = dev.has(sycl::aspect::ext_intel_matrix);
        caps.has_fp16  = true;
        caps.has_int8  = caps.has_bf16;
        // Gaudi 3: FP8 (check via aspect if available)
        caps.has_fp8   = false;  // TODO: Gaudi3 fp8 aspect check
        caps.has_flash_attn = caps.has_bf16;
        caps.shared_mem_kb  = (int)(
            dev.get_info<sycl::info::device::local_mem_size>() / 1024);
        caps.n_devices = 1;  // simplified
    } catch (...) {}
#else
    caps.n_devices = 1;
#endif
    return caps;
}

int device_count() {
    int n = 1;
#if defined(AMP_BACKEND_CUDA)
    cudaGetDeviceCount(&n);
#elif defined(AMP_BACKEND_HIP)
    hipGetDeviceCount(&n);
#endif
    return n > 0 ? n : 1;
}

RuntimeInfo detect_runtime() {
    RuntimeInfo rt;
#if defined(AMP_BACKEND_CUDA)
    {
        int ver = 0;
        cudaRuntimeGetVersion(&ver);
        int maj = ver / 1000, min = (ver % 1000) / 10;
        rt.runtime_ver = std::to_string(maj) + "." + std::to_string(min);
    }
    {
        int drv = 0;
        cuDriverGetVersion(&drv);
        int maj = drv / 1000, min = (drv % 1000) / 10;
        rt.driver_ver = std::to_string(maj) + "." + std::to_string(min);
    }
#elif defined(AMP_BACKEND_HIP)
    {
        int ver = 0;
        hipRuntimeGetVersion(&ver);
        // HIP version encoding: major*10000000 + minor*100000 + patch
        int maj = ver / 10000000, min = (ver / 100000) % 100;
        rt.runtime_ver = std::to_string(maj) + "." + std::to_string(min);
        rt.driver_ver  = rt.runtime_ver;
    }
#elif defined(AMP_BACKEND_SYCL)
    try {
        rt.runtime_ver = AMP::detail::default_q()
            .get_device().get_info<sycl::info::device::driver_version>();
        rt.driver_ver = rt.runtime_ver;
    } catch (...) {
        rt.runtime_ver = rt.driver_ver = "unknown";
    }
#else
    rt.runtime_ver = "0.0";
    rt.driver_ver  = "0.0";
#endif
    return rt;
}

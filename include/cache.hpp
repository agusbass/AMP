// cache.hpp - Module 3: persistent compile cache
#pragma once
#include <string>
#include <vector>
#include <optional>

namespace AMP {

// Describes the compilation target for fingerprinting.
// runtime_ver is included so cache entries automatically invalidate on driver upgrades.
struct CompileGraph {
    std::string op;           // "matmul", "attn", ...
    std::vector<int> shape;
    std::string dtype;        // "fp32", "bf16", "fp16"
    std::string arch;         // vendor:device:runtime_ver — unique key per-hardware+driver
    std::string tune_cfg;     // serialized config (from autotuner)
    std::string runtime_ver;  // "12.6" (CUDA) | "6.1" (ROCm) | "" → ignored
};

class CacheStore {
public:
    explicit CacheStore(const std::string& root_dir);

    // Return cached binary; nullopt if miss, checksum mismatch, or stale format.
    std::optional<std::vector<char>> get(const std::string& key);

    // Store with a binary header (magic + version + FNV checksum).
    void put(const std::string& key, const std::vector<char>& binary);

    // SHA256-based content-addressable key: SHA256(graph fields) → 16 hex chars.
    static std::string fingerprint(const CompileGraph& g);

    size_t size() const;

private:
    std::string root_;
};

} // namespace AMP

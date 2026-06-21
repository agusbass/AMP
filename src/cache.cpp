// cache.cpp - persistent cache implementation with binary header
#include "cache.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace AMP {
namespace fs = std::filesystem;

// =====================================================================
// Minimal SHA256 (RFC 6234) — pure C++, no openssl dep.
// =====================================================================
namespace {

struct SHA256 {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint64_t bit_len = 0;
    uint8_t buf[64];
    size_t buf_len = 0;
    static uint32_t rot(uint32_t x, int n) { return (x>>n)|(x<<(32-n)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
            0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
            0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
            0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
            0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
            0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
            0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
            0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
            0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i=0;i<16;i++)
            w[i]=(p[i*4]<<24)|(p[i*4+1]<<16)|(p[i*4+2]<<8)|p[i*4+3];
        for (int i=16;i<64;i++){
            uint32_t s0=rot(w[i-15],7)^rot(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=rot(w[i-2],17)^rot(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i=0;i<64;i++){
            uint32_t S1=rot(e,6)^rot(e,11)^rot(e,25);
            uint32_t ch=(e&f)^(~e&g);
            uint32_t t1=hh+S1+ch+K[i]+w[i];
            uint32_t S0=rot(a,2)^rot(a,13)^rot(a,22);
            uint32_t mj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+mj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    void update(const uint8_t* d, size_t n) {
        bit_len += n*8;
        while (n) {
            size_t k=std::min((size_t)(64-buf_len),n);
            std::memcpy(buf+buf_len,d,k);
            buf_len+=k; d+=k; n-=k;
            if (buf_len==64){ block(buf); buf_len=0; }
        }
    }
    std::string hexdigest() {
        uint64_t bl=bit_len;
        uint8_t pad=0x80;
        update(&pad,1);
        while (buf_len!=56){ uint8_t z=0; update(&z,1); }
        uint8_t lb[8];
        for (int i=0;i<8;i++) lb[7-i]=(bl>>(i*8))&0xff;
        update(lb,8);
        std::ostringstream o;
        for (int i=0;i<8;i++) o<<std::hex<<std::setw(8)<<std::setfill('0')<<h[i];
        return o.str();
    }
};

// =====================================================================
// Binary entry header — bump kVersion when the format changes.
// Old entries automatically become a miss → re-tune (no crash).
// =====================================================================
static constexpr char     kMagic[4] = {'U','X','L','C'};
static constexpr uint32_t kVersion  = 1;

#pragma pack(push, 1)
struct BinHeader {
    char     magic[4];
    uint32_t version;
    uint32_t payload_crc;   // FNV-1a-32 over the payload
    uint32_t payload_len;
};  // 16 bytes, packed
#pragma pack(pop)

// FNV-1a-32: fast, better than XOR, sufficient for an integrity check.
uint32_t fnv1a32(const char* d, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) h = (h ^ (uint8_t)d[i]) * 16777619u;
    return h;
}

} // anon

// =====================================================================
// CacheStore impl
// =====================================================================

std::string CacheStore::fingerprint(const CompileGraph& g) {
    SHA256 s;
    std::ostringstream o;
    // All fields + runtime_ver go into the key → driver upgrade = cache miss
    o << g.op << "|" << g.dtype << "|" << g.arch << "|"
      << g.tune_cfg << "|" << g.runtime_ver << "|";
    for (int d : g.shape) o << d << ",";
    auto str = o.str();
    s.update((const uint8_t*)str.data(), str.size());
    return s.hexdigest().substr(0, 16);
}

CacheStore::CacheStore(const std::string& root) : root_(root) {
    fs::create_directories(root_);
}

std::optional<std::vector<char>> CacheStore::get(const std::string& key) {
    auto p = fs::path(root_) / (key + ".bin");
    if (!fs::exists(p)) return std::nullopt;

    std::ifstream f(p, std::ios::binary);
    BinHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f) return std::nullopt;

    // Validate magic + version (reject stale entries or corrupted files)
    if (std::memcmp(hdr.magic, kMagic, 4) != 0 ||
        hdr.version != kVersion) {
        return std::nullopt;
    }

    std::vector<char> bin(hdr.payload_len);
    f.read(bin.data(), hdr.payload_len);
    if (!f) return std::nullopt;

    // Validate payload integrity
    if (fnv1a32(bin.data(), bin.size()) != hdr.payload_crc)
        return std::nullopt;

    return bin;
}

void CacheStore::put(const std::string& key, const std::vector<char>& bin) {
    BinHeader hdr{};
    std::memcpy(hdr.magic, kMagic, 4);
    hdr.version     = kVersion;
    hdr.payload_crc = fnv1a32(bin.data(), bin.size());
    hdr.payload_len = static_cast<uint32_t>(bin.size());

    auto p = fs::path(root_) / (key + ".bin");
    // Atomic write via temp file: prevent partial-write corruption
    auto tmp = fs::path(root_) / (key + ".tmp");
    {
        std::ofstream f(tmp, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        f.write(bin.data(), bin.size());
    }
    fs::rename(tmp, p);
}

size_t CacheStore::size() const {
    size_t n = 0;
    for (auto& e : fs::directory_iterator(root_))
        if (e.path().extension() == ".bin") n++;
    return n;
}

} // namespace AMP

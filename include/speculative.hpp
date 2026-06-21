// speculative.hpp - Speculative Decoding engine
//
// References:
//   "Fast Inference from Transformers via Speculative Decoding"
//   Leviathan et al. 2022 / Chen et al. 2023
//
// Algorithm (rejection sampling):
//   1. Draft model generates `k` speculative tokens in 1 small fwd pass
//   2. Target model verifies all `k` tokens at once in 1 fwd pass
//   3. For each position i:
//      - If q(x) >= p(x): accept (q=draft prob, p=target prob)
//      - Else: accept with prob p(x)/q(x), reject with prob 1-p(x)/q(x)
//   4. After rejection: sample 1 token from adjusted distribution
//
// Supported variants:
//   - Standard speculative decoding (separate draft model)
//   - Self-speculative / Medusa: draft heads on top of the target model
//   - Greedy verification (temperature=0): accept if argmax matches
//
// AMP implementation:
//   SpecDecodeEngine only implements the STATISTICS (acceptance logic).
//   Model execution (draft forward, target forward) is performed by the
//   caller via callback. This decouples the scheduler from the model framework.
#pragma once
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <optional>

namespace AMP {

// ================================================================
// Config
// ================================================================

struct SpecConfig {
    int   num_speculative_tokens = 5;    // k: number of draft tokens per step
    float temperature            = 1.0f; // for sampling from adjusted dist
    bool  greedy_verify          = false;// true = only compare argmax
    int   vocab_size             = 32000;
    unsigned seed                = 42;
};

// ================================================================
// Token distribution (output of model forward pass)
// ================================================================

struct TokenDist {
    std::vector<float> logits;  // raw logits, shape [vocab_size]
    int                sampled; // token already sampled by the caller

    // Lazy softmax: computed when needed
    mutable std::vector<float> probs_;
    mutable bool probs_computed_ = false;

    const std::vector<float>& probs(float temperature = 1.0f) const {
        if (!probs_computed_) {
            probs_.resize(logits.size());
            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum = 0.0f;
            for (size_t i = 0; i < logits.size(); ++i) {
                probs_[i] = std::exp((logits[i] - max_l) / std::max(temperature, 1e-6f));
                sum += probs_[i];
            }
            for (auto& p : probs_) p /= sum;
            probs_computed_ = true;
        }
        return probs_;
    }
};

// ================================================================
// Step result
// ================================================================

struct SpecStepResult {
    std::vector<int> accepted_tokens; // accepted tokens (0..k)
    int              bonus_token = -1;// 1 bonus token from adjusted dist (-1 = none)
    int              n_accepted  = 0; // number of accepted draft tokens
    double           acceptance_rate = 0.0; // running average

    // Total tokens produced this step = n_accepted + (bonus_token >= 0 ? 1 : 0)
    int total_new() const {
        return n_accepted + (bonus_token >= 0 ? 1 : 0);
    }
};

// ================================================================
// SpecDecodeEngine
// ================================================================

class SpecDecodeEngine {
public:
    explicit SpecDecodeEngine(const SpecConfig& cfg);

    // ---- Verify + accept/reject ----
    // draft_tokens: k tokens from the draft model [0..k-1]
    // draft_dists : draft model distribution for each position [0..k-1]
    // target_dists: target model distribution for positions [0..k] (k+1 fwd)
    //               target_dists[k] = distribution for the position after all drafts
    //
    // Returns: accepted tokens + 1 bonus token
    SpecStepResult verify(
        const std::vector<int>&      draft_tokens,
        const std::vector<TokenDist>& draft_dists,
        const std::vector<TokenDist>& target_dists);

    // ---- Greedy verification (temperature=0) ----
    // Faster: only compares draft argmax vs target argmax
    SpecStepResult verify_greedy(
        const std::vector<int>&      draft_tokens,
        const std::vector<TokenDist>& target_dists);

    // ---- Stats ----
    struct Stats {
        std::atomic<uint64_t> steps{0};
        std::atomic<uint64_t> draft_tokens{0};
        std::atomic<uint64_t> accepted_tokens{0};
        std::atomic<uint64_t> rejected_tokens{0};

        double acceptance_rate() const {
            uint64_t d = draft_tokens.load();
            return d > 0 ? (double)accepted_tokens.load() / d : 0.0;
        }
        double speedup_estimate() const {
            // Expected speedup = (E[accepted] + 1) / 1 target fwd
            // E[accepted] = k * acceptance_rate
            return acceptance_rate() * (double)draft_tokens.load() /
                   std::max((uint64_t)1, steps.load()) + 1.0;
        }
    };
    const Stats& stats() const { return stats_; }

    const SpecConfig& config() const { return cfg_; }

private:
    SpecConfig  cfg_;
    std::mt19937 rng_;
    Stats       stats_;

    int sample_from_adjusted(const std::vector<float>& p_target,
                              const std::vector<float>& p_draft,
                              float temperature);
    int sample_from(const std::vector<float>& probs, float temperature);
};

// ================================================================
// Medusa-style multi-head speculative (self-speculative)
// ================================================================
// Medusa adds several "draft heads" on top of the main model.
// Each head predicts token offset t+1, t+2, ..., t+k.
// No separate model needed — a single forward pass suffices.

struct MedusaConfig {
    int  num_heads             = 4;    // number of draft heads
    int  num_speculative_tokens = 4;   // same as num_heads
    bool tree_attention         = true; // true = tree-based verification
    float acceptance_threshold  = 0.0f; // 0 = standard rejection sampling
};

// Medusa verification: processes the acceptance tree
// candidates: each head produces a top-1 token
// target_logits: target model output for each node in the tree
struct MedusaResult {
    std::vector<int> accepted_path;
    int              total_accepted = 0;  // REQUIRED: int is not auto-initialized in C++
};

MedusaResult medusa_verify(
    const MedusaConfig&              cfg,
    const std::vector<int>&          candidates,   // [num_heads] top-1 per head
    const std::vector<TokenDist>&    target_dists, // [num_heads+1]
    float temperature,
    std::mt19937& rng);

} // namespace AMP

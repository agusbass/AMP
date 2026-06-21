// speculative.cpp - rejection sampling + Medusa verification
#include "speculative.hpp"
#include <numeric>
#include <cassert>

namespace AMP {

SpecDecodeEngine::SpecDecodeEngine(const SpecConfig& cfg)
    : cfg_(cfg), rng_(cfg.seed) {}

// ================================================================
// Standard rejection sampling verification
// ================================================================
SpecStepResult SpecDecodeEngine::verify(
    const std::vector<int>&      draft_tokens,
    const std::vector<TokenDist>& draft_dists,
    const std::vector<TokenDist>& target_dists)
{
    assert(draft_tokens.size() == draft_dists.size());
    assert(target_dists.size() == draft_tokens.size() + 1);

    SpecStepResult result;
    stats_.steps.fetch_add(1);
    stats_.draft_tokens.fetch_add(draft_tokens.size());

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    int k = (int)draft_tokens.size();

    for (int i = 0; i < k; ++i) {
        int x = draft_tokens[i];
        const auto& p_t = target_dists[i].probs(cfg_.temperature);
        const auto& p_d = draft_dists[i].probs(cfg_.temperature);

        float pt = (x < (int)p_t.size()) ? p_t[x] : 0.0f;
        float pd = (x < (int)p_d.size()) ? p_d[x] : 1e-9f;

        float accept_prob = std::min(1.0f, pt / (pd + 1e-9f));

        if (uniform(rng_) <= accept_prob) {
            // Accept
            result.accepted_tokens.push_back(x);
            result.n_accepted++;
            stats_.accepted_tokens.fetch_add(1);
        } else {
            // Reject at position i → sample bonus from adjusted dist
            stats_.rejected_tokens.fetch_add(k - i);  // all remaining are also rejected

            // Adjusted distribution: max(0, p_target - p_draft) normalized
            const auto& adj_p = target_dists[i].probs(cfg_.temperature);
            const auto& adj_d = draft_dists[i].probs(cfg_.temperature);
            result.bonus_token = sample_from_adjusted(adj_p, adj_d, cfg_.temperature);
            goto done;
        }
    }

    // All k accepted: sample bonus from target dist at position k
    result.bonus_token = sample_from(
        target_dists[k].probs(cfg_.temperature), cfg_.temperature);

done:
    // Update running acceptance rate
    uint64_t d = stats_.draft_tokens.load();
    uint64_t a = stats_.accepted_tokens.load();
    result.acceptance_rate = d > 0 ? (double)a / d : 0.0;
    return result;
}

// ================================================================
// Greedy verification (temperature=0)
// ================================================================
SpecStepResult SpecDecodeEngine::verify_greedy(
    const std::vector<int>&      draft_tokens,
    const std::vector<TokenDist>& target_dists)
{
    SpecStepResult result;
    stats_.steps.fetch_add(1);
    stats_.draft_tokens.fetch_add(draft_tokens.size());
    int k = (int)draft_tokens.size();

    for (int i = 0; i < k; ++i) {
        // Target argmax
        const auto& logits = target_dists[i].logits;
        int target_tok = (int)(std::max_element(logits.begin(), logits.end())
                                - logits.begin());
        if (draft_tokens[i] == target_tok) {
            result.accepted_tokens.push_back(draft_tokens[i]);
            result.n_accepted++;
            stats_.accepted_tokens.fetch_add(1);
        } else {
            // Reject: use target argmax as bonus
            stats_.rejected_tokens.fetch_add(k - i);
            result.bonus_token = target_tok;
            return result;
        }
    }

    // All accepted: bonus = target argmax at k
    const auto& logits_k = target_dists[k].logits;
    result.bonus_token = (int)(std::max_element(logits_k.begin(), logits_k.end())
                                - logits_k.begin());
    return result;
}

// ================================================================
// Sampling helpers
// ================================================================

int SpecDecodeEngine::sample_from_adjusted(
    const std::vector<float>& p_target,
    const std::vector<float>& p_draft,
    float temperature)
{
    // Adjusted distribution: p'(x) = max(0, p_t(x) - p_d(x)), then normalize
    int V = (int)p_target.size();
    std::vector<float> adj(V);
    float sum = 0.0f;
    for (int i = 0; i < V; ++i) {
        float pd = (i < (int)p_draft.size()) ? p_draft[i] : 0.0f;
        adj[i] = std::max(0.0f, p_target[i] - pd);
        sum += adj[i];
    }
    if (sum < 1e-9f) {
        // Degenerate: sample uniformly
        std::uniform_int_distribution<int> uni(0, V-1);
        return uni(rng_);
    }
    for (auto& a : adj) a /= sum;
    return sample_from(adj, 1.0f);  // already temperature-adjusted
}

int SpecDecodeEngine::sample_from(const std::vector<float>& probs,
                                    float /*temperature*/)
{
    // Categorical sampling via CDF
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float r = u(rng_);
    float cum = 0.0f;
    for (int i = 0; i < (int)probs.size(); ++i) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return (int)probs.size() - 1;
}

// ================================================================
// Medusa verification (tree attention)
// ================================================================

MedusaResult medusa_verify(
    const MedusaConfig&              cfg,
    const std::vector<int>&          candidates,
    const std::vector<TokenDist>&    target_dists,
    float temperature,
    std::mt19937& rng)
{
    MedusaResult result{};  // value-initialize: total_accepted=0

    // Without tree attention: linear chain acceptance (simplified Medusa)
    // With tree: would need tree construction + parallel scoring.
    // This implements the linear fallback (Medusa-1).
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    for (int i = 0; i < cfg.num_heads && i < (int)candidates.size(); ++i) {
        int cand = candidates[i];
        if (i >= (int)target_dists.size()) break;

        const auto& p_t = target_dists[i].probs(temperature);
        float pt = (cand < (int)p_t.size()) ? p_t[cand] : 0.0f;

        // Medusa uses threshold-based acceptance (not rejection sampling)
        bool accept = (cfg.acceptance_threshold > 0.0f)
            ? (pt >= cfg.acceptance_threshold)
            : (uniform(rng) <= pt);  // standard rejection sampling

        if (accept) {
            result.accepted_path.push_back(cand);
            result.total_accepted++;
        } else {
            break;
        }
    }

    return result;
}

} // namespace AMP

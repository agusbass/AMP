// autotune_sh.hpp - Successive Halving autotuner
// Algorithm: budget B is split into r rounds. Each round, evaluate all configs
// at a small budget, drop the worst half, double the budget per survivor.
// Result: instead of O(N * full_reps), it becomes O(N log N) with proportional reps.
#pragma once
#include <vector>
#include <functional>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace AMP {

template<typename Cfg>
struct SHResult {
    Cfg best;
    double best_score;       // seconds/op or ms — lower is better
    int total_evals;         // number of actual measurements (vs grid search)
};

// `measure(cfg, n_reps)` -> average time (lower=better)
template<typename Cfg>
SHResult<Cfg> successive_halving(
    const std::vector<Cfg>& candidates,
    std::function<double(const Cfg&, int)> measure,
    int min_reps = 1, int max_reps = 32, double eta = 2.0)
{
    if (candidates.empty())
        throw std::invalid_argument("successive_halving: candidates is empty");

    struct Item { Cfg cfg; double score; };
    std::vector<Item> alive;
    alive.reserve(candidates.size());
    for (auto& c : candidates) alive.push_back({c, 0.0});

    int total_evals = 0;
    int reps = min_reps;
    while (alive.size() > 1) {
        for (auto& it : alive) {
            it.score = measure(it.cfg, reps);
            total_evals += reps;
        }
        std::sort(alive.begin(), alive.end(),
                  [](const Item& a, const Item& b){ return a.score < b.score; });
        size_t keep = std::max<size_t>(1, alive.size() / (size_t)eta);
        alive.resize(keep);
        reps = std::min(max_reps, (int)(reps * eta));
        if (reps >= max_reps && alive.size() == 1) break;
    }
    // final precision on the survivor
    alive[0].score = measure(alive[0].cfg, max_reps);
    total_evals += max_reps;
    return {alive[0].cfg, alive[0].score, total_evals};
}

// Grid search baseline (for comparison)
template<typename Cfg>
SHResult<Cfg> grid_search(
    const std::vector<Cfg>& candidates,
    std::function<double(const Cfg&, int)> measure,
    int reps = 32)
{
    Cfg best{}; double best_s = 1e18; int evals = 0;
    for (auto& c : candidates) {
        double s = measure(c, reps);
        evals += reps;
        if (s < best_s) { best_s = s; best = c; }
    }
    return {best, best_s, evals};
}

} // namespace AMP

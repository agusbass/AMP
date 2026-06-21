// scheduler.cpp - Continuous Batching + Chunked Prefill
#include "scheduler.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace AMP {

Scheduler::Scheduler(const SchedulerConfig& cfg) : cfg_(cfg) {
    // Default page_bytes: for LLaMA-7B style (32 heads × 128 dim × BF16 × 2 (K+V))
    int pb = cfg_.page_bytes > 0
        ? cfg_.page_bytes
        : cfg_.kv_page_size_tokens * 32 * 128 * 2 * 2;  // estimate
    kv_pool_ = std::make_unique<PagePoolFast>(
        cfg_.max_pages_total, (size_t)pb, /*n_shards=*/16);
}

Scheduler::~Scheduler() = default;

// ---- Add request ----
int Scheduler::add_request(std::vector<int> prompt_tokens,
                            int max_new_tokens, float temperature,
                            int eos_token_id)
{
    auto seq = std::make_unique<Sequence>();
    seq->id             = next_id_.fetch_add(1);
    seq->prompt_tokens  = std::move(prompt_tokens);
    seq->max_new_tokens = max_new_tokens;
    seq->temperature    = temperature;
    seq->eos_token_id   = eos_token_id;
    seq->state          = SeqState::WAITING;
    seq->enqueue_time   = std::chrono::steady_clock::now();
    seq->prefill_chunk_size = cfg_.max_prefill_tokens_per_step;

    int id = seq->id;
    total_requests_.fetch_add(1);

    std::lock_guard g(mtx_);
    if ((int)waiting_.size() >= cfg_.max_waiting_seqs)
        throw std::runtime_error("Scheduler waiting queue full");
    waiting_.push_back(std::move(seq));
    return id;
}

void Scheduler::abort_request(int seq_id) {
    std::lock_guard g(mtx_);
    auto abort_in = [&](auto& vec) {
        auto it = std::find_if(vec.begin(), vec.end(),
            [seq_id](const auto& s){ return s->id == seq_id; });
        if (it != vec.end()) {
            free_pages(**it);
            (*it)->state = SeqState::ABORTED;
            vec.erase(it);
            return true;
        }
        return false;
    };
    abort_in(waiting_);
    abort_in(running_);
    abort_in(swapped_);
}

// ---- Pages needed ----
int Scheduler::pages_needed_for(const Sequence& s, int n_tokens) const {
    int total = s.total_tokens() + n_tokens;
    return (total + cfg_.kv_page_size_tokens - 1) / cfg_.kv_page_size_tokens;
}

bool Scheduler::try_alloc_pages(Sequence& s, int n_new_tokens) {
    int needed = pages_needed_for(s, n_new_tokens);
    int have   = (int)s.kv_page_ids.size();
    if (needed <= have) return true;

    int extra = needed - have;
    // thread_hash: use sequence id for shard affinity
    auto new_pages = kv_pool_->alloc(extra, s.id);
    if ((int)new_pages.size() < extra) return false;

    for (int p : new_pages) s.kv_page_ids.push_back(p);
    return true;
}

void Scheduler::free_pages(Sequence& s) {
    if (!s.kv_page_ids.empty()) {
        kv_pool_->free_pages(s.kv_page_ids);
        s.kv_page_ids.clear();
    }
}

void Scheduler::preempt_sequence(Sequence& s) {
    // RECOMPUTE mode: free KV pages, send back to waiting
    free_pages(s);
    s.state         = SeqState::WAITING;
    s.prefill_cursor = 0;  // re-prefill from scratch
    s.output_tokens.clear();
    preemptions_.fetch_add(1);
}

// ---- Promote waiting → running ----
void Scheduler::promote_waiting() {
    while (!waiting_.empty() &&
           (int)running_.size() < cfg_.max_seqs_running) {
        auto& front = waiting_.front();

        // Allocate at least 1 page for prefill start
        if (!try_alloc_pages(*front, cfg_.kv_page_size_tokens)) break;

        front->state = SeqState::PREFILL;
        running_.push_back(std::move(front));
        waiting_.pop_front();
    }
}

// ---- schedule() — called before each forward pass ----
ScheduledBatch Scheduler::schedule() {
    std::lock_guard g(mtx_);
    ScheduledBatch batch;

    // 1. Promote waiting requests (FCFS)
    promote_waiting();

    int tokens_budget = cfg_.max_batch_tokens;

    // 2. Build batch entries
    // Priority: PREFILL first (to unblock KV slots), then DECODE
    std::vector<Sequence*> prefill_seqs, decode_seqs;
    for (auto& s : running_) {
        if (s->state == SeqState::PREFILL) prefill_seqs.push_back(s.get());
        else if (s->state == SeqState::DECODE) decode_seqs.push_back(s.get());
    }

    // Chunked prefill: each prefill seq gets a chunk
    for (Sequence* s : prefill_seqs) {
        if (tokens_budget <= 0) break;
        int chunk = std::min({
            s->remaining_prefill(),
            cfg_.max_prefill_tokens_per_step,
            tokens_budget
        });
        if (chunk <= 0) continue;

        if (!try_alloc_pages(*s, chunk)) {
            // OOM: preempt lowest-priority decode seq
            if (!decode_seqs.empty()) {
                preempt_sequence(*decode_seqs.back());
                decode_seqs.pop_back();
                batch.n_preempted++;
                if (!try_alloc_pages(*s, chunk)) continue;  // still OOM, skip
            } else { continue; }
        }

        BatchEntry e;
        e.seq           = s;
        e.n_tokens      = chunk;
        e.is_prefill    = true;
        e.prefill_start = s->prefill_cursor;
        batch.entries.push_back(e);
        batch.total_tokens += chunk;
        batch.n_prefill++;
        tokens_budget -= chunk;
    }

    // Decode: 1 token per seq
    for (Sequence* s : decode_seqs) {
        if (tokens_budget <= 0) break;
        if (!try_alloc_pages(*s, 1)) {
            preempt_sequence(*s);
            batch.n_preempted++;
            continue;
        }
        BatchEntry e;
        e.seq        = s;
        e.n_tokens   = 1;
        e.is_prefill = false;
        e.prefill_start = 0;
        batch.entries.push_back(e);
        batch.total_tokens += 1;
        batch.n_decode++;
        tokens_budget -= 1;
    }

    return batch;
}

// ---- step_complete() — called after forward pass ----
void Scheduler::step_complete(const ScheduledBatch& batch,
                               const std::vector<int>& new_tokens)
{
    std::lock_guard g(mtx_);
    int decode_idx = 0;

    for (const auto& e : batch.entries) {
        Sequence* s = e.seq;

        if (e.is_prefill) {
            s->prefill_cursor += e.n_tokens;
            if (s->remaining_prefill() <= 0) {
                s->state = SeqState::DECODE;
            }
        } else {
            // Decode: append new token
            int tok = (decode_idx < (int)new_tokens.size())
                ? new_tokens[decode_idx++] : 0;
            s->output_tokens.push_back(tok);
            tokens_generated_.fetch_add(1);
            kv_pool_->touch(s->kv_page_ids);

            bool done = (int)s->output_tokens.size() >= s->max_new_tokens ||
                        (s->eos_token_id >= 0 && tok == s->eos_token_id);
            if (done) {
                s->state = SeqState::DONE;
                if (completion_cb_) completion_cb_(s->id, s->output_tokens);
                completed_.fetch_add(1);
            }
        }
    }

    // Remove done/aborted from running
    running_.erase(
        std::remove_if(running_.begin(), running_.end(),
            [](const auto& s){ return s->is_done(); }),
        running_.end());

    // KV leak scan (free stale pages from aborted/timed-out seqs)
    kv_pool_->leak_scan(/*ttl_ms=*/30000);
}

// ---- Stats ----
Scheduler::Stats Scheduler::get_stats() const {
    Stats s;
    s.total_requests        = total_requests_.load();
    s.completed             = completed_.load();
    s.preemptions           = preemptions_.load();
    s.total_tokens_generated = tokens_generated_.load();
    {
        std::lock_guard g(mtx_);
        s.n_waiting = (int)waiting_.size();
        s.n_running = (int)running_.size();
        s.n_swapped = (int)swapped_.size();
    }
    auto ps = kv_pool_->stats();
    s.kv_pages_used = ps.in_use;
    s.kv_pages_free = ps.total - ps.in_use;
    return s;
}

} // namespace AMP

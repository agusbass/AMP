// scheduler.hpp - Continuous Batching + Chunked Prefill scheduler
//
// References: Orca (Yu et al. 2022), vLLM (Kwon et al. 2023)
//
// Design:
//   - Iteration-level scheduling: each step, the scheduler decides
//     which requests run and how many tokens are processed.
//   - Chunked prefill: long prompts are split into small chunks,
//     interleaved with decode steps -> latency fairness.
//   - Paged KV cache: via PagePoolFast, pages allocated per step.
//   - Preemption: low-priority requests can be swapped out on OOM.
//
// State machine per request:
//   WAITING → PREFILL → DECODE → DONE
//                  ↓         ↓
//               SWAPPED ← ← ←    (preempted, KV pages freed)
//                  ↓
//              WAITING           (re-queued for retry)
#pragma once
#include "portable.hpp"
#include "pool_fast.hpp"
#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>
#include <optional>
#include <chrono>
#include <atomic>
#include <mutex>

namespace AMP {

// ================================================================
// Request & Sequence
// ================================================================

enum class SeqState {
    WAITING,   // waiting in queue, no KV allocation yet
    PREFILL,   // currently prefilling (chunked or full)
    DECODE,    // prefill done, currently autoregressive decoding
    SWAPPED,   // KV pages evicted due to OOM, waiting for re-prefill
    DONE,      // finished (max_tokens reached or EOS)
    ABORTED,   // cancelled by caller
};

struct Sequence {
    int         id;
    std::string prompt;
    std::vector<int> prompt_tokens;
    std::vector<int> output_tokens;
    int         max_new_tokens  = 256;
    float       temperature     = 1.0f;
    float       top_p           = 1.0f;
    int         eos_token_id    = -1;   // -1 = no EOS

    SeqState    state           = SeqState::WAITING;

    // KV cache pages allocated from PagePoolFast
    std::vector<int> kv_page_ids;
    int  kv_pages_used          = 0;  // how many blocks are filled

    // Chunked prefill state
    int  prefill_cursor         = 0;  // next token to be prefilled
    int  prefill_chunk_size     = 512;

    // Timestamp for priority/timeout
    std::chrono::steady_clock::time_point enqueue_time;

    bool is_done() const {
        return state == SeqState::DONE || state == SeqState::ABORTED;
    }
    int total_tokens() const {
        return (int)prompt_tokens.size() + (int)output_tokens.size();
    }
    int remaining_prefill() const {
        return (int)prompt_tokens.size() - prefill_cursor;
    }
};

// ================================================================
// Scheduler config
// ================================================================

struct SchedulerConfig {
    // KV cache
    int    kv_page_size_tokens = 16;   // tokens per KV page
    int    max_pages_total     = 2048; // total KV pages pool
    int    page_bytes          = 0;    // 0 = auto from head_dim + dtype

    // Batching limits per step
    int    max_batch_tokens    = 4096; // max tokens processed per step
    int    max_seqs_running    = 256;  // max simultaneous requests
    int    max_waiting_seqs    = 2048; // max queue size

    // Chunked prefill
    bool   enable_chunked_prefill = true;
    int    max_prefill_tokens_per_step = 2048;

    // Preemption policy
    enum class PreemptMode { RECOMPUTE, SWAP } preempt_mode = PreemptMode::RECOMPUTE;

    // Priority: FCFS or shortest-remaining (SRT)
    enum class Policy { FCFS, SRT } policy = Policy::FCFS;
};

// ================================================================
// Batch: output scheduler per step
// ================================================================

struct BatchEntry {
    Sequence*   seq;
    int         n_tokens;     // tokens processed this step
    bool        is_prefill;   // true = prefill chunk, false = decode
    int         prefill_start;// start index in prompt (for prefill)
};

struct ScheduledBatch {
    std::vector<BatchEntry> entries;
    int total_tokens     = 0;
    int n_prefill        = 0;
    int n_decode         = 0;
    int n_preempted      = 0;  // requests evicted this step
};

// ================================================================
// Scheduler
// ================================================================

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& cfg);
    ~Scheduler();

    // ---- Request lifecycle ----
    // Add request to the waiting queue. Returns sequence id.
    int add_request(std::vector<int> prompt_tokens,
                    int max_new_tokens = 256,
                    float temperature  = 1.0f,
                    int eos_token_id   = -1);

    // Cancel a request that is running or waiting
    void abort_request(int seq_id);

    // ---- Scheduling step ----
    // Call before each forward pass.
    // Returns the batch to execute this step.
    ScheduledBatch schedule();

    // ---- Post-step update ----
    // Called after the forward pass finishes with the generated tokens.
    // new_tokens[i] = token generated for entries[i] (decode only).
    void step_complete(const ScheduledBatch& batch,
                       const std::vector<int>& new_tokens);

    // ---- Stats ----
    // Plain POD snapshot — copyable, no atomics
    struct Stats {
        uint64_t total_requests       = 0;
        uint64_t completed            = 0;
        uint64_t preemptions          = 0;
        uint64_t total_tokens_generated = 0;
        int n_waiting     = 0;
        int n_running     = 0;
        int n_swapped     = 0;
        int kv_pages_used = 0;
        int kv_pages_free = 0;
    };
    Stats get_stats() const;

    // Callback invoked when a sequence finishes
    using CompletionCallback = std::function<void(int seq_id,
                                                   const std::vector<int>& tokens)>;
    void set_completion_callback(CompletionCallback cb) {
        completion_cb_ = std::move(cb);
    }

private:
    SchedulerConfig cfg_;
    std::unique_ptr<PagePoolFast> kv_pool_;

    // Queues (mutex-protected)
    mutable std::mutex mtx_;
    std::deque<std::unique_ptr<Sequence>> waiting_;
    std::vector<std::unique_ptr<Sequence>> running_;
    std::vector<std::unique_ptr<Sequence>> swapped_;

    std::atomic<int> next_id_{0};
    CompletionCallback completion_cb_;
    mutable std::atomic<uint64_t> total_requests_{0};
    mutable std::atomic<uint64_t> completed_{0};
    mutable std::atomic<uint64_t> preemptions_{0};
    mutable std::atomic<uint64_t> tokens_generated_{0};

    // ---- Internal helpers ----
    int pages_needed_for(const Sequence& s, int n_tokens) const;
    bool try_alloc_pages(Sequence& s, int n_new_tokens);
    void free_pages(Sequence& s);
    void preempt_sequence(Sequence& s);
    void promote_waiting();
};

} // namespace AMP

#pragma once

#include <array>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "Planner.h"

/*  Plans on a worker thread, handed to the audio thread without a lock.

    Measured on an M2, ONNX Runtime on four CPU threads:

        flow      ~120 ms   paid once per map position
        decode    ~560 ms   paid once per sounding note

    The split is the whole reason this class is shaped the way it is. The flow
    predictor never sees MIDI -- verified bit-exact -- so a trajectory is cached
    against the position that produced it and every note at that position is a
    decode and nothing more. A four-note chord costs one flow and four decodes
    rather than four of each.

    What it does *not* fix is that four decodes is still about 2.2 s. The real
    answer is block-streaming the decoder, which measures 41 ms per 16384
    samples and would put four voices comfortably inside real time; the graph
    already takes a window_start input for exactly that. Until then the voice
    count is a user-visible trade against reported latency rather than a number
    picked quietly.

    The handoff is deliberately not a shared_ptr swap: releasing the last
    reference would free about a megabyte on the audio thread, and freeing is
    not real-time safe. The worker owns every plan it has made in a small ring
    per voice and publishes a raw pointer; the audio thread never touches a
    reference count.
*/

namespace atlasflow {

class PlanService
{
public:
    static constexpr int kMaxVoices = 4;

    PlanService (const std::string& flowPath, const std::string& decoderPath)
        : planner (flowPath, decoderPath)
    {
        worker = std::thread ([this] { warmUp(); run(); });
    }

    ~PlanService()
    {
        {
            std::lock_guard<std::mutex> lock (mutex);
            quitting = true;
        }
        signal.notify_all();
        if (worker.joinable())
            worker.join();
    }

    /** Ask for audio for one voice. Newest-wins per voice: a drag across the
        map emits positions far faster than plans complete, and queueing them
        made the Python runtime chase the pointer long after the user stopped. */
    void request (int voiceIndex, const std::array<float, kAtlasDims>& normalized,
                  int note, bool restart, float temperature = 0.0f, uint64_t seed = 20260822)
    {
        if (voiceIndex < 0 || voiceIndex >= kMaxVoices)
            return;
        {
            std::lock_guard<std::mutex> lock (mutex);
            auto& slot = pending[(size_t) voiceIndex];

            // A morph must never bury an unserved note-on. The slot is
            // newest-wins by design -- a drag emits positions far faster than
            // plans complete -- but "newest" applied blindly meant the morph
            // issued on the next block overwrote the note-on before the worker
            // ever saw it, and the note simply never started. Take the newer
            // position, keep the restart.
            const bool keepRestart = slot.live && slot.restart && ! restart;
            slot.position = normalized;
            slot.note = keepRestart ? slot.note : note;
            slot.restart = keepRestart ? true : restart;
            slot.temperature = temperature;
            slot.seed = seed;
            slot.generation = generation[(size_t) voiceIndex].load (std::memory_order_relaxed);
            slot.live = true;
        }
        signal.notify_one();
    }

    /** Drop anything pending for this voice and make any plan still in flight
        arrive stale. Called on note-off: without it, releasing a key before its
        plan landed left the plan to arrive afterwards and start sounding, which
        reads as a note that refuses to stop. */
    void cancel (int voiceIndex)
    {
        if (voiceIndex < 0 || voiceIndex >= kMaxVoices)
            return;
        generation[(size_t) voiceIndex].fetch_add (1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock (mutex);
        pending[(size_t) voiceIndex].live = false;
    }

    /** Whether this voice has work queued or running. The processor uses it to
        issue one morph at a time instead of one per audio block, which is the
        difference between following the path and drowning the planner. */
    bool busyFor (int voiceIndex) const
    {
        if (voiceIndex < 0 || voiceIndex >= kMaxVoices)
            return false;
        std::lock_guard<std::mutex> lock (mutex);
        return pending[(size_t) voiceIndex].live
            || (working.load (std::memory_order_relaxed)
                && servingVoice.load (std::memory_order_relaxed) == voiceIndex);
    }

    /** Audio thread, lock-free: a plan for this note at the position the bank
        was filled for, or nullptr.

        This is what makes zero reported latency possible. Decoding at note-on
        costs 0.4-1.0 s and cannot be hidden; decoding *before* note-on costs
        nothing at all, because the trained range is only 36 pitches and the
        trajectory behind them is shared. A note-on then becomes what it
        already was for sustain -- reading PCM that exists -- and the instrument
        becomes playable rather than merely renderable.
    */
    const Plan* fromBank (int note) const noexcept
    {
        if (note < 0 || note >= 128)
            return nullptr;
        return bank[(size_t) note].load (std::memory_order_acquire);
    }

    /** Which position the bank currently holds, so the processor can tell
        whether a note-on can be served instantly. */
    uint64_t bankIdentity() const noexcept { return bankStamp.load (std::memory_order_acquire); }

    /** Point the bank at a position without waiting for a note.

        Called when the map settles, so selecting a sound starts rendering it
        straight away instead of at the first key press. Cheap and idempotent:
        if it is already the cached position nothing happens. */
    void prime (const std::array<float, kAtlasDims>& normalized, float temperature, uint64_t seed)
    {
        std::lock_guard<std::mutex> lock (mutex);
        primePosition = snap (normalized);
        primeTemperature = temperature;
        primeSeed = seed;
        primeWanted = true;
        signal.notify_one();
    }

    /** Hint the fill order: pitches near what was last played come first, so a
        player is covered long before the full range is. */
    void setBankCentre (int note) noexcept { bankCentre.store (note, std::memory_order_relaxed); }

    int bankReady() const noexcept { return bankCount.load (std::memory_order_relaxed); }

    /** Audio thread: tell the owner what this voice can still read from.

        The cache is the sole owner of every plan it has produced, and an LRU
        that evicted blindly would free one a sustaining voice is still playing
        -- which is exactly how this crashed once the cache made publication
        fast enough for the old fixed-size ring to wrap. */
    void publishHeld (int voiceIndex, const Plan* a, const Plan* b, const Plan* c) noexcept
    {
        if (voiceIndex < 0 || voiceIndex >= kMaxVoices)
            return;
        held[(size_t) voiceIndex * 3 + 0].store (a, std::memory_order_relaxed);
        held[(size_t) voiceIndex * 3 + 1].store (b, std::memory_order_relaxed);
        held[(size_t) voiceIndex * 3 + 2].store (c, std::memory_order_relaxed);
    }

    /** Audio thread: the newest finished plan for this voice, or nullptr. */
    const Plan* tryAcquire (int voiceIndex) noexcept
    {
        if (voiceIndex < 0 || voiceIndex >= kMaxVoices)
            return nullptr;
        return ready[(size_t) voiceIndex].exchange (nullptr, std::memory_order_acquire);
    }

    bool lastWasRestart (int voiceIndex) const noexcept
    {
        return deliveredRestart[(size_t) voiceIndex].load (std::memory_order_relaxed);
    }

    /** Block until nothing is pending or in flight. Offline rendering only: a
        bounce runs far faster than real time, so a background plan would never
        land and the render would come out silent. There is no deadline to miss
        offline, so waiting is both safe and the only correct answer -- and it
        is forbidden from a real-time callback. */
    void waitUntilSettled (int timeoutMilliseconds = 120000)
    {
        const auto deadline = std::chrono::steady_clock::now()
                            + std::chrono::milliseconds (timeoutMilliseconds);
        for (;;)
        {
            {
                std::lock_guard<std::mutex> lock (mutex);
                bool anyPending = working.load (std::memory_order_relaxed);
                for (const auto& slot : pending)
                    anyPending = anyPending || slot.live;
                if (! anyPending)
                    return;
            }
            if (std::chrono::steady_clock::now() >= deadline)
                return;
            std::this_thread::sleep_for (std::chrono::milliseconds (1));
        }
    }

    bool busy() const noexcept { return working.load (std::memory_order_relaxed); }
    bool warmedUp() const noexcept { return warm.load (std::memory_order_relaxed); }
    unsigned hits() const noexcept { return cacheHits.load (std::memory_order_relaxed); }
    unsigned misses() const noexcept { return cacheMisses.load (std::memory_order_relaxed); }
    double lastFlowMilliseconds() const noexcept { return flowMs.load (std::memory_order_relaxed); }
    double lastDecodeMilliseconds() const noexcept { return decodeMs.load (std::memory_order_relaxed); }
    bool trajectoryWasReused() const noexcept { return reused.load (std::memory_order_relaxed); }
    const AtlasGeometry& geometry() const { return planner.geometry(); }

private:
    struct Request
    {
        std::array<float, kAtlasDims> position {};
        int note = 60;
        bool restart = true;
        float temperature = 0.0f;
        uint64_t seed = 20260822;
        unsigned generation = 0;
        bool live = false;
    };

    static bool samePosition (const std::array<float, kAtlasDims>& a,
                              const std::array<float, kAtlasDims>& b) noexcept
    {
        for (size_t i = 0; i < a.size(); ++i)
            if (std::abs (a[i] - b[i]) > 1.0e-6f)
                return false;
        return true;
    }

    /** Snap a request onto a grid before planning.

        Consolidation, not precision loss: the atlas projection already pulls
        any query onto a hull of real presets, so sub-grid motion mostly lands
        in the same place anyway. Snapping means a walking path re-plans when it
        crosses a cell rather than continuously, and -- the point -- that the
        same stretch of path on the next lap asks for a key that is already in
        the cache.
    */
    static std::array<float, kAtlasDims> snap (std::array<float, kAtlasDims> p) noexcept
    {
        constexpr float cells = 48.0f;
        for (auto& v : p)
            v = std::round (v * cells) / cells;
        return p;
    }

    struct CacheKey
    {
        std::array<float, kAtlasDims> position {};
        int note = -1;
        bool restart = true;
        float temperature = 0.0f;
        uint64_t seed = 0;

        bool operator== (const CacheKey& other) const noexcept
        {
            return note == other.note && restart == other.restart
                && temperature == other.temperature && seed == other.seed
                && samePosition (position, other.position);
        }
    };

    /** Run each graph once on throwaway input before any note arrives.

        ONNX Runtime allocates its arenas and picks its kernels on the first
        call, which costs roughly half as much again as a warm one -- measured
        837 ms cold against 393 ms warm for a decode. Paying that at load time
        instead of on the user's first note is both better behaved and what
        lets the reported latency be sized from the warm figure rather than the
        worst one.
    */
    void warmUp()
    {
        try
        {
            std::array<float, kAtlasDims> centre {};
            auto trajectory = planner.planTrajectory (centre, true);
            const auto started = std::chrono::steady_clock::now();
            (void) planner.decode (trajectory.values, 60);
            const double decode = std::chrono::duration<double> (
                std::chrono::steady_clock::now() - started).count();

            // Measured, not assumed. The same binary runs 1.8x slower under
            // Rosetta than natively, and a latency sized from the native figure
            // would be a deadline the translated build cannot meet -- which is
            // the configuration a host with Rosetta enabled actually gives us.
            warmFlowSeconds.store (trajectory.milliseconds / 1000.0, std::memory_order_relaxed);
            warmDecodeSeconds.store (decode, std::memory_order_relaxed);

            // Keep the trajectory rather than throwing it away: the idle loop
            // starts filling the bank from it immediately, so the stretch
            // between loading the plugin and playing the first note -- which
            // is otherwise dead time -- is spent on the work that would
            // otherwise land on the first note.
            cached = std::move (trajectory);
            cachedPosition = centre;
            cachedRestart = true;
            cachedTemperature = 0.0f;
            cachedSeed = 20260822;
            haveTrajectory = true;

            warm.store (true, std::memory_order_release);
        }
        catch (const std::exception&)
        {
            warm.store (true, std::memory_order_release);   // never leave a caller waiting
        }
    }

public:
    /** Wait for the load-time warm-up, so the first latency report is sized
        from this machine rather than from a constant. Called from
        prepareToPlay, which is not a real-time context. */
    bool waitUntilWarm (int timeoutMilliseconds = 20000)
    {
        const auto deadline = std::chrono::steady_clock::now()
                            + std::chrono::milliseconds (timeoutMilliseconds);
        while (! warm.load (std::memory_order_acquire))
        {
            if (std::chrono::steady_clock::now() >= deadline)
                return false;
            std::this_thread::sleep_for (std::chrono::milliseconds (2));
        }
        return true;
    }

    double measuredFlowSeconds() const noexcept
    {
        return warmFlowSeconds.load (std::memory_order_relaxed);
    }
    double measuredDecodeSeconds() const noexcept
    {
        return warmDecodeSeconds.load (std::memory_order_relaxed);
    }

private:

    void run()
    {
        for (;;)
        {
            int voiceIndex = -1;
            Request job;
            {
                std::unique_lock<std::mutex> lock (mutex);
                // Wake for work, or to carry on filling the bank. A note the
                // player has asked for always wins over one they have not.
                signal.wait_for (lock, std::chrono::milliseconds (bankIncomplete() ? 1 : 200),
                                 [this] { return anyPendingLocked() || quitting; });
                if (quitting)
                    return;
                if (! anyPendingLocked())
                {
                    const bool wantPrime = primeWanted;
                    auto position = primePosition;
                    const float temperature = primeTemperature;
                    const uint64_t seed = primeSeed;
                    primeWanted = false;
                    lock.unlock();

                    if (wantPrime && ! (haveTrajectory
                                        && samePosition (cachedPosition, position)
                                        && cachedTemperature == temperature
                                        && cachedSeed == seed))
                    {
                        try
                        {
                            cached = planner.planTrajectory (position, true, nullptr, temperature, seed);
                            cachedPosition = position;
                            cachedRestart = true;
                            cachedTemperature = temperature;
                            cachedSeed = seed;
                            haveTrajectory = true;
                        }
                        catch (const std::exception&) { }
                    }
                    fillBankStep();
                    continue;
                }
                for (int v = 0; v < kMaxVoices; ++v)
                    if (pending[(size_t) v].live) { voiceIndex = v; job = pending[(size_t) v]; break; }
                if (voiceIndex < 0)
                    continue;
                pending[(size_t) voiceIndex].live = false;
            }

            servingVoice.store (voiceIndex, std::memory_order_relaxed);
            working.store (true, std::memory_order_relaxed);
            try
            {
                job.position = snap (job.position);

                // A drawn path walks the same ground on the way back, and a
                // progression revisits the same chords, so most of what gets
                // asked for after the first lap has already been rendered.
                const CacheKey key { job.position, job.note, job.restart,
                                     job.temperature, job.seed };
                if (auto hit = lookUp (key))
                {
                    if (job.generation == generation[(size_t) voiceIndex].load (std::memory_order_relaxed))
                    {
                        deliveredRestart[(size_t) voiceIndex].store (job.restart, std::memory_order_relaxed);
                        ready[(size_t) voiceIndex].store (hit.get(), std::memory_order_release);
                    }
                    cacheHits.fetch_add (1, std::memory_order_relaxed);
                    working.store (false, std::memory_order_relaxed);
                    continue;
                }
                cacheMisses.fetch_add (1, std::memory_order_relaxed);
                // One flow per position; every note at that position is a
                // decode and nothing more.
                // Temperature and seed change the trajectory, so they are part
                // of the cache key: reusing across them would silently ignore a
                // control the user just moved.
                const bool hit = haveTrajectory
                              && samePosition (cachedPosition, job.position)
                              && cachedRestart == job.restart
                              && cachedTemperature == job.temperature
                              && cachedSeed == job.seed;
                if (! hit)
                {
                    cached = planner.planTrajectory (job.position, job.restart, nullptr,
                                                     job.temperature, job.seed);
                    cachedPosition = job.position;
                    cachedRestart = job.restart;
                    cachedTemperature = job.temperature;
                    cachedSeed = job.seed;
                    haveTrajectory = true;
                    flowMs.store (cached.milliseconds, std::memory_order_relaxed);
                }
                reused.store (hit, std::memory_order_relaxed);

                auto plan = std::make_shared<Plan> (planner.render (cached, job.note));
                decodeMs.store (plan->decodeMilliseconds, std::memory_order_relaxed);
                store (key, plan);

                // Drop it on the floor if the note was released while this was
                // being computed.
                if (job.generation == generation[(size_t) voiceIndex].load (std::memory_order_relaxed))
                {
                    deliveredRestart[(size_t) voiceIndex].store (job.restart, std::memory_order_relaxed);
                    ready[(size_t) voiceIndex].store (plan.get(), std::memory_order_release);
                }
            }
            catch (const std::exception&)
            {
                // A failed plan must not take the audio thread with it; the
                // voice keeps playing whatever it already had.
            }
            working.store (false, std::memory_order_relaxed);
        }
    }

    bool bankIncomplete() const noexcept
    {
        return haveTrajectory && bankCount.load (std::memory_order_relaxed) < kBankPitches;
    }

    /** Decode one more pitch for the current trajectory, nearest the centre
        first. One pitch per wake so a real request never waits long behind it. */
    void fillBankStep()
    {
        if (! haveTrajectory)
            return;

        const uint64_t identity = positionIdentity (cachedPosition, cachedTemperature, cachedSeed);
        if (identity != bankStamp.load (std::memory_order_relaxed))
        {
            // The map moved: everything in the bank describes a sound that is
            // no longer selected, so retire it rather than serve it.
            for (auto& slot : bank)
                slot.store (nullptr, std::memory_order_release);
            bankOwned.clear();
            bankCount.store (0, std::memory_order_relaxed);
            bankStamp.store (identity, std::memory_order_release);
        }
        if (bankCount.load (std::memory_order_relaxed) >= kBankPitches)
            return;

        const int centre = std::clamp (bankCentre.load (std::memory_order_relaxed), kLowNote, kHighNote);
        for (int radius = 0; radius <= kBankPitches; ++radius)
        {
            for (int sign : { 1, -1 })
            {
                const int note = centre + radius * sign;
                if (note < kLowNote || note > kHighNote)
                    continue;
                if (bank[(size_t) note].load (std::memory_order_relaxed) != nullptr)
                    continue;

                try
                {
                    auto plan = std::make_shared<Plan> (planner.render (cached, note));
                    bank[(size_t) note].store (plan.get(), std::memory_order_release);
                    bankOwned.push_back (std::move (plan));
                    bankCount.fetch_add (1, std::memory_order_relaxed);
                }
                catch (const std::exception&) { }
                return;
            }
        }
    }

public:
    /** The bank's identity for a position; the processor needs the same hash
        to know whether what is banked is what is selected. */
    static uint64_t identityFor (const std::array<float, kAtlasDims>& p,
                                 float temperature, uint64_t seed) noexcept
    { return positionIdentity (snap (p), temperature, seed); }

private:
    static uint64_t positionIdentity (const std::array<float, kAtlasDims>& p,
                                      float temperature, uint64_t seed) noexcept
    {
        uint64_t h = 1469598103934665603ull ^ seed;
        auto mix = [&h] (float v)
        {
            uint32_t bits;
            std::memcpy (&bits, &v, sizeof (bits));
            h = (h ^ bits) * 1099511628211ull;
        };
        for (float v : p) mix (v);
        mix (temperature);
        return h | 1ull;   // never zero, so "no bank" stays distinguishable
    }

    std::shared_ptr<Plan> lookUp (const CacheKey& key)
    {
        for (size_t i = 0; i < cache.size(); ++i)
            if (cache[i].first == key)
            {
                auto found = cache[i].second;
                // Most-recently-used to the front, so the entries that survive
                // eviction are the ones a lap keeps coming back to.
                std::rotate (cache.begin(), cache.begin() + (long) i, cache.begin() + (long) i + 1);
                return found;
            }
        return nullptr;
    }

    bool isHeld (const Plan* plan) const noexcept
    {
        if (plan == nullptr)
            return false;
        for (const auto& slot : held)
            if (slot.load (std::memory_order_relaxed) == plan)
                return true;
        return false;
    }

    void store (const CacheKey& key, std::shared_ptr<Plan> plan)
    {
        cache.insert (cache.begin(), { key, std::move (plan) });

        // Evict from the least-recently-used end, but never something a voice
        // can still read from. A sustaining note holds its plan for as long as
        // the key is down, which is far longer than any fixed-size ring covers.
        while (cache.size() > kCacheEntries)
        {
            auto victim = cache.end();
            for (auto it = cache.end(); it != cache.begin(); )
            {
                --it;
                if (! isHeld (it->second.get())) { victim = it; break; }
            }
            if (victim == cache.end())
                break;              // everything is in use; grow for now
            cache.erase (victim);
        }
    }

    bool anyPendingLocked() const
    {
        for (const auto& slot : pending)
            if (slot.live) return true;
        return false;
    }

    // A plan is about 940 KB, so this is roughly 22 MB of rendered audio.
    static constexpr size_t kCacheEntries = 24;
    // The trained range, which is the whole of what a bank must cover.
    static constexpr int kLowNote = 36, kHighNote = 71;
    static constexpr int kBankPitches = kHighNote - kLowNote + 1;

    Planner planner;
    std::thread worker;

    mutable std::mutex mutex;
    std::condition_variable signal;
    std::array<Request, kMaxVoices> pending {};
    bool quitting = false;
    std::array<float, kAtlasDims> primePosition {};
    float primeTemperature = 0.0f;
    uint64_t primeSeed = 20260822;
    bool primeWanted = false;

    // Worker-only state.
    Trajectory cached;
    std::array<float, kAtlasDims> cachedPosition {};
    bool cachedRestart = true;
    float cachedTemperature = -1.0f;
    uint64_t cachedSeed = 0;
    bool haveTrajectory = false;


    std::array<std::atomic<const Plan*>, kMaxVoices> ready {};
    std::array<std::atomic<bool>, kMaxVoices> deliveredRestart {};
    std::atomic<bool> working { false };
    std::atomic<int> servingVoice { -1 };
    std::array<std::atomic<unsigned>, kMaxVoices> generation {};
    std::atomic<bool> warm { false };
    std::atomic<double> warmFlowSeconds { 0.0 };
    std::atomic<double> warmDecodeSeconds { 0.0 };
    std::atomic<bool> reused { false };
    std::atomic<double> flowMs { 0.0 };
    std::atomic<double> decodeMs { 0.0 };
    std::vector<std::pair<CacheKey, std::shared_ptr<Plan>>> cache;
    std::atomic<unsigned> cacheHits { 0 }, cacheMisses { 0 };
    std::array<std::atomic<const Plan*>, kMaxVoices * 3> held {};
    std::array<std::atomic<const Plan*>, 128> bank {};
    std::vector<std::shared_ptr<Plan>> bankOwned;
    std::atomic<uint64_t> bankStamp { 0 };
    std::atomic<int> bankCentre { 60 };
    std::atomic<int> bankCount { 0 };
};

}  // namespace atlasflow

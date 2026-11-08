#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "Planner.h"

/*  A port of the playback half of AtlasFlowLiveSession.

    This is what turns a five-second plan into an instrument. The model renders
    one canonical note; holding a key longer than that works by time-warping the
    mature part of the render rather than generating more, which is why sustain
    costs nothing while morphing costs a plan.

    Three crossfades, all equal-power, all from the Python:

      - the sustain loop, 0.2 s, which splices the end of the mature region back
        to its start for as long as a key is held;
      - the morph, 0.5-5 s, between the plan that is sounding and a new one at a
        new map position, read at the same cursor so the note does not restart;
      - the release, 0.1 s, from wherever playback was into the learned release
        tail at 2.6 s.

    Nothing here runs a network. Everything expensive already happened in
    Planner, which is why this is safe to call from an audio callback and the
    planner is not.
*/

namespace atlasflow {

inline constexpr float kSustainLoopStartSeconds = 1.2f;
inline constexpr float kSustainLoopEndMarginSeconds = 0.2f;
inline constexpr float kSustainLoopCrossfadeSeconds = 0.2f;
inline constexpr float kReleaseCrossfadeSeconds = 0.1f;
inline constexpr float kNoteChangeCrossfadeSeconds = 0.05f;
inline constexpr float kDefaultMorphSeconds = 2.0f;

class Voice
{
public:
    enum class State { idle, heldSustain, release };

    Voice()
    {
        loopStart = std::max (kNoteOnSample + 1,
                              (int) std::lround (kSustainLoopStartSeconds * kSampleRate));
        loopEnd = std::min (kNoteOffSample - 1,
                            kNoteOffSample - (int) std::lround (kSustainLoopEndMarginSeconds * kSampleRate));
        loopCrossfade = (int) std::lround (kSustainLoopCrossfadeSeconds * kSampleRate);
    }

    /** Hand the voice a freshly computed plan.

        `restart` means a note-on or a pitch change: playback jumps back to the
        attack and the crossfade is short, because splicing two different
        pitches at a shared cursor is a glitch rather than a morph. Otherwise it
        is a morph: the incoming plan is read from the *same* cursor as the one
        sounding, so the note keeps its place in its own envelope.
    */
    void accept (const Plan* plan, bool restart, float morphSeconds = kDefaultMorphSeconds)
    {
        if (active.plan == nullptr)
        {
            active = { plan, kNoteOnSample };
            state = State::heldSustain;
            return;
        }
        if (incoming.plan != nullptr)
        {
            // A drag emits positions faster than plans complete; hold the newest
            // and let the running crossfade finish rather than stacking layers.
            queued = plan;
            queuedRestart = restart;
            queuedMorphSeconds = morphSeconds;
            return;
        }
        startIncoming (plan, restart, morphSeconds);
    }

    void noteOff()
    {
        if (active.plan == nullptr || state == State::idle || state == State::release)
            return;

        // Capture a short slice of what is sounding *before* jumping, then
        // crossfade it into the release region; otherwise the jump from the
        // sustain loop to 2.6 s is a step discontinuity.
        const int count = (int) std::lround (kReleaseCrossfadeSeconds * kSampleRate);
        releaseTail.assign ((size_t) count, 0.0f);
        renderMix (releaseTail.data(), count);
        releaseTailCursor = 0;

        state = State::release;
        queued = nullptr;
        active.cursor = kNoteOffSample;
        if (incoming.plan != nullptr)
            incoming.cursor = kNoteOffSample;
    }

    bool active_() const { return state != State::idle; }
    State currentState() const { return state; }

    /** Every plan this voice could still read from. The owner uses it to avoid
        freeing something under the audio thread's feet. */
    void heldPlans (const Plan* out[3]) const noexcept
    {
        out[0] = active.plan;
        out[1] = incoming.plan;
        out[2] = queued;
    }

    /** Fill `count` samples. Never allocates, never blocks, never runs a net. */
    void render (float* out, int count, float velocity = 1.0f)
    {
        std::fill (out, out + count, 0.0f);
        if (active.plan == nullptr)
            return;

        renderMix (out, count);

        if (! releaseTail.empty())
        {
            const int remaining = (int) releaseTail.size() - releaseTailCursor;
            const int amount = std::min (count, remaining);
            const float span = (float) std::max ((int) releaseTail.size() - 1, 1);
            for (int i = 0; i < amount; ++i)
            {
                const float phase = (float) (releaseTailCursor + i) / span * 1.57079632679f;
                out[i] = releaseTail[(size_t) (releaseTailCursor + i)] * std::cos (phase)
                       + out[i] * std::sin (phase);
            }
            releaseTailCursor += amount;
            if (releaseTailCursor >= (int) releaseTail.size())
                releaseTail.clear();
        }

        for (int i = 0; i < count; ++i)
            out[i] *= velocity;

        if (state == State::release
            && active.plan != nullptr
            && active.cursor >= kRenderSamples
            && (incoming.plan == nullptr || incoming.cursor >= kRenderSamples))
        {
            active = {};
            incoming = {};
            state = State::idle;
        }
    }

private:
    struct Layer
    {
        const Plan* plan = nullptr;   // owned by PlanService, never by the voice
        int cursor = 0;
    };

    void startIncoming (const Plan* plan, bool restart, float morphSeconds)
    {
        if (active.plan == nullptr)
        {
            active = { plan, kNoteOnSample };
            state = State::heldSustain;
            return;
        }
        const int cursor = restart ? kNoteOnSample : active.cursor;
        incoming = { plan, cursor };
        const float seconds = restart ? kNoteChangeCrossfadeSeconds : morphSeconds;
        fadeSamples = std::max (1, (int) std::lround (seconds * kSampleRate));
        fadeCursor = 0;
    }

    /** Straight playback to the end of the render; used during release. */
    void readLinear (Layer& layer, float* out, int count)
    {
        std::fill (out, out + count, 0.0f);
        const int stop = std::min (layer.cursor + count, kRenderSamples);
        const int available = std::max (0, stop - layer.cursor);
        for (int i = 0; i < available; ++i)
            out[i] = layer.plan->audio[(size_t) (layer.cursor + i)];
        layer.cursor = stop;
    }

    /** The sustain loop: play to the end of the mature region, then equal-power
        crossfade back to its start, forever. */
    void readHeld (Layer& layer, float* out, int count)
    {
        const int fadeStart = loopEnd - loopCrossfade;
        int written = 0;
        while (written < count)
        {
            if (layer.cursor >= loopEnd)
                layer.cursor = loopStart + loopCrossfade;

            if (layer.cursor < fadeStart)
            {
                const int amount = std::min (count - written, fadeStart - layer.cursor);
                for (int i = 0; i < amount; ++i)
                    out[written + i] = layer.plan->audio[(size_t) (layer.cursor + i)];
                layer.cursor += amount;
                written += amount;
                continue;
            }

            const int amount = std::min (count - written, loopEnd - layer.cursor);
            const int offset = layer.cursor - fadeStart;
            for (int i = 0; i < amount; ++i)
            {
                // numpy linspace over `amount` points, inclusive of both ends.
                const float t = amount > 1
                    ? (float) (offset + i) / (float) loopCrossfade
                    : (float) offset / (float) loopCrossfade;
                const float phase = std::clamp (t, 0.0f, 1.0f) * 1.57079632679f;
                const float old = layer.plan->audio[(size_t) (layer.cursor + i)];
                const float fresh = layer.plan->audio[(size_t) (loopStart + offset + i)];
                out[written + i] = old * std::cos (phase) + fresh * std::sin (phase);
            }
            layer.cursor += amount;
            written += amount;
        }
    }

    void readLayer (Layer& layer, float* out, int count)
    {
        if (state == State::release) readLinear (layer, out, count);
        else                         readHeld (layer, out, count);
    }

    void renderMix (float* out, int count)
    {
        if (active.plan == nullptr)
        {
            std::fill (out, out + count, 0.0f);
            return;
        }

        scratchLeft.resize ((size_t) count);
        readLayer (active, scratchLeft.data(), count);

        if (incoming.plan == nullptr)
        {
            std::copy (scratchLeft.begin(), scratchLeft.begin() + count, out);
            return;
        }

        scratchRight.resize ((size_t) count);
        readLayer (incoming, scratchRight.data(), count);

        const float span = (float) std::max (fadeSamples - 1, 1);
        for (int i = 0; i < count; ++i)
        {
            const float phase = std::clamp ((float) (fadeCursor + i) / span, 0.0f, 1.0f) * 1.57079632679f;
            out[i] = scratchLeft[(size_t) i] * std::cos (phase)
                   + scratchRight[(size_t) i] * std::sin (phase);
        }
        fadeCursor += count;

        if (fadeCursor >= fadeSamples)
        {
            active = incoming;
            incoming = {};
            fadeCursor = 0;
            if (queued != nullptr && state != State::release)
            {
                const Plan* next = queued;
                queued = nullptr;
                startIncoming (next, queuedRestart, queuedMorphSeconds);
            }
        }
    }

    Layer active, incoming;
    const Plan* queued = nullptr;
    bool queuedRestart = false;
    float queuedMorphSeconds = kDefaultMorphSeconds;

    State state = State::idle;
    int fadeCursor = 0, fadeSamples = 1;
    std::vector<float> releaseTail;
    int releaseTailCursor = 0;
    int loopStart = 0, loopEnd = 0, loopCrossfade = 0;
    std::vector<float> scratchLeft, scratchRight;
};

}  // namespace atlasflow

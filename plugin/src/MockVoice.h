#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "Atlas.h"

namespace atlasflow {

/** The placeholder instrument.

    Every anchor on the map is a fixed mix of three basic waveforms -- sine,
    saw, square -- chosen deterministically from the preset index. Moving
    across the map crossfades those mixes. There is no neural network here and
    no attempt to sound like the trained model; the point is that the map is
    navigable and audibly continuous, so the MIDI path, the voice allocation
    and the UI can be finished and judged before the ONNX decoder lands.

    Replacing this with the real decoder should not require touching the
    processor or the editor: the contract is "give me a note and a blend over
    presets, I will fill a buffer".
*/
struct PresetTimbre
{
    float sine, saw, square;
};

/** Deterministic per-preset mix. Stable across runs and across machines, so a
    session saved today opens the same tomorrow. */
inline PresetTimbre timbreFor (int preset) noexcept
{
    // Cheap integer hash (splitmix-ish); any stable scramble would do.
    uint32_t h = (uint32_t) preset * 2654435761u + 1013904223u;
    auto next = [&h]() noexcept
    {
        h ^= h >> 16; h *= 2246822519u;
        h ^= h >> 13; h *= 3266489917u;
        h ^= h >> 16;
        return (float) (h & 0xffffffu) / (float) 0xffffffu;
    };
    PresetTimbre t { next(), next(), next() };
    // Bias toward sine so the map is not uniformly buzzy, then normalise.
    t.sine = t.sine * 0.6f + 0.4f;
    t.saw *= 0.8f;
    t.square *= 0.6f;
    const float sum = t.sine + t.saw + t.square;
    t.sine /= sum; t.saw /= sum; t.square /= sum;
    return t;
}

/** One monophonic voice.

    The real model is monophonic too -- a chord is N independent voices summed
    -- so this mirrors the shape the decoder will need rather than inventing a
    polyphonic structure that would have to be unwound later.
*/
class MockVoice
{
public:
    /** The trained model only knows MIDI 36-71. Folding by octaves keeps the
        pitch class correct; clamping (what the Python runtime does today)
        silently retunes the note, which is wrong in a DAW where the host sends
        whatever the user played. */
    static int foldToRange (int note) noexcept
    {
        while (note < 36) note += 12;
        while (note > 71) note -= 12;
        return note;
    }

    void prepare (double sampleRateIn) noexcept
    {
        sampleRate = sampleRateIn;
        reset();
    }

    void reset() noexcept
    {
        phase = 0.0; env = 0.0f; stage = Stage::idle;
        note = -1; velocity = 0.0f;
    }

    void start (int midiNote, float vel, uint32_t stamp) noexcept
    {
        note = foldToRange (midiNote);
        velocity = vel;
        increment = 440.0 * std::pow (2.0, (note - 69) / 12.0) / sampleRate;
        phase = 0.0;
        stage = Stage::attack;
        age = stamp;
    }

    void release() noexcept { if (stage != Stage::idle) stage = Stage::release; }

    bool active() const noexcept { return stage != Stage::idle; }
    int  currentNote() const noexcept { return note; }
    uint32_t stamp() const noexcept { return age; }

    /** Add this voice into the buffer.

        The waveform balance arrives as three per-sample ramps rather than one
        value per block: a drag across the map moves the blend continuously,
        and holding it constant for a block puts audible steps on a sustained
        pad. The caller owns the ramps so that every voice reads the same one.
    */
    void render (float* out, int numSamples,
                 const float* mixSine, const float* mixSaw, const float* mixSquare) noexcept
    {
        if (stage == Stage::idle)
            return;

        // A pad attack, deliberately slow-ish: the trained model needs 1.2 s to
        // reach its sustain loop, so a mock with a 2 ms attack would make the
        // instrument feel quite different from what ships later.
        const float attackRate  = (float) (1.0 / (0.25 * sampleRate));
        const float releaseRate = (float) (1.0 / (0.60 * sampleRate));

        for (int i = 0; i < numSamples; ++i)
        {
            if (stage == Stage::attack)
            {
                env += attackRate;
                if (env >= 1.0f) { env = 1.0f; stage = Stage::sustain; }
            }
            else if (stage == Stage::release)
            {
                env -= releaseRate;
                if (env <= 0.0f) { env = 0.0f; stage = Stage::idle; break; }
            }

            const float t = (float) phase;
            const float dt = (float) increment;
            const float s = std::sin (6.28318530718f * t);
            const float w = mixSine[i] * s
                          + mixSaw[i] * saw (t, dt)
                          + mixSquare[i] * square (t, dt);

            out[i] += w * env * velocity * 0.22f;

            phase += increment;
            if (phase >= 1.0) phase -= 1.0;
        }
    }

private:
    enum class Stage { idle, attack, sustain, release };

    /** polyBLEP band-limiting. Naive saw/square alias badly enough to be
        distracting while judging the map, and this is twenty lines. */
    static float blep (float t, float dt) noexcept
    {
        if (t < dt)            { t /= dt;      return t + t - t * t - 1.0f; }
        if (t > 1.0f - dt)     { t = (t - 1.0f) / dt; return t * t + t + t + 1.0f; }
        return 0.0f;
    }

    static float saw (float t, float dt) noexcept
    {
        return (2.0f * t - 1.0f) - blep (t, dt);
    }

    static float square (float t, float dt) noexcept
    {
        float v = t < 0.5f ? 1.0f : -1.0f;
        v += blep (t, dt);
        float t2 = t + 0.5f; if (t2 >= 1.0f) t2 -= 1.0f;
        v -= blep (t2, dt);
        return v;
    }

    double sampleRate = 44100.0;
    double phase = 0.0, increment = 0.0;
    float env = 0.0f, velocity = 0.0f;
    int note = -1;
    uint32_t age = 0;
    Stage stage = Stage::idle;
};

}  // namespace atlasflow

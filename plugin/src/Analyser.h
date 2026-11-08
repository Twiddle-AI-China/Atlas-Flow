#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <vector>

#include <juce_dsp/juce_dsp.h>

namespace atlasflow {

/** The audio thread's end of the scope: a tap, not a queue.

    The scopes want the *most recent* window every frame, not every sample ever
    written, so this is a circular buffer plus a monotonic count rather than a
    FIFO. The reader copies the tail; if it is unlucky enough to read a sample
    the audio thread is overwriting at that instant, one frame of one scope has
    one wrong sample in it, and the next frame does not. The buffer holds 740 ms
    at 44.1 kHz, which is twenty frames of slack against a reader that stalls.

    Nothing here blocks or allocates, which is the only requirement the audio
    thread actually imposes.
*/
class ScopeTap
{
public:
    static constexpr int kSize = 1 << 15;        // 32768 samples
    static constexpr long long kMask = kSize - 1;

    void write (const float* samples, int count) noexcept
    {
        // Masked, not modulo: the counter is signed, so % would compile to a
        // real integer division per sample on the audio thread. The size is a
        // power of two and the counter only ever grows from zero, so the mask
        // is exact.
        const auto start = written.load (std::memory_order_relaxed);
        for (int i = 0; i < count; ++i)
            buffer[(size_t) ((start + i) & kMask)] = samples[i];
        written.store (start + count, std::memory_order_release);
    }

    /** Copy the newest `count` samples, oldest first. */
    void latest (float* destination, int count) const noexcept
    {
        const auto end = written.load (std::memory_order_acquire);
        for (int i = 0; i < count; ++i)
        {
            const auto at = end - count + i;
            destination[i] = at < 0 ? 0.0f : buffer[(size_t) (at & kMask)];
        }
    }

    long long samplesSeen() const noexcept { return written.load (std::memory_order_acquire); }

private:
    std::array<float, kSize> buffer {};
    std::atomic<long long> written { 0 };
};

/** A Web Audio AnalyserNode, reproduced.

    The desktop instrument's scopes are written against an AnalyserNode and
    every constant in them -- the 1.2 exponent on the spectrogram, the 0.15
    floor on the waterfall, where the loudness traces sit in the pane -- is
    tuned to what that node returns. Reimplementing the spectrum "properly"
    here, with a different window or a different dB range, would move every one
    of those and the panes would need retuning to look the same. So this is the
    specification, not an improvement on it:

      - fftSize 2048, as buildChain() sets it
      - smoothingTimeConstant 0.45, as buildChain() sets it
      - Blackman window, a0 = 0.42
      - magnitudes divided by fftSize
      - mapped over [-100 dB, -30 dB], the node's own default range

    The one thing not reproduced is the quantisation to bytes. getByteFrequency
    Data() returns 0-255 and every view divides by 255 again; keeping floats
    skips a rounding step that nothing downstream can see.
*/
class Analyser
{
public:
    static constexpr int kFftOrder = 11;                 // 2048
    static constexpr int kFftSize = 1 << kFftOrder;
    static constexpr int kBins = kFftSize / 2;
    static constexpr float kSmoothing = 0.45f;
    static constexpr float kMinDecibels = -100.0f;
    static constexpr float kMaxDecibels = -30.0f;

    Analyser() : fft (kFftOrder)
    {
        for (int i = 0; i < kFftSize; ++i)
        {
            const double phase = juce::MathConstants<double>::twoPi * i / kFftSize;
            window[(size_t) i] = (float) (0.42 - 0.5 * std::cos (phase) + 0.08 * std::cos (2.0 * phase));
        }
    }

    /** Pull one window from the tap and fold it into the smoothed spectrum. */
    void update (const ScopeTap& tap)
    {
        tap.latest (time.data(), kFftSize);

        std::copy (time.begin(), time.end(), scratch.begin());
        std::fill (scratch.begin() + kFftSize, scratch.end(), 0.0f);
        for (int i = 0; i < kFftSize; ++i)
            scratch[(size_t) i] *= window[(size_t) i];

        fft.performFrequencyOnlyForwardTransform (scratch.data(), true);

        for (int bin = 0; bin < kBins; ++bin)
        {
            // performFrequencyOnlyForwardTransform leaves magnitudes; Web Audio
            // divides them by fftSize before taking the logarithm.
            const float magnitude = scratch[(size_t) bin] / (float) kFftSize;
            smoothed[(size_t) bin] = kSmoothing * smoothed[(size_t) bin]
                                   + (1.0f - kSmoothing) * magnitude;
            const float db = juce::Decibels::gainToDecibels (smoothed[(size_t) bin], -200.0f);
            level[(size_t) bin] = juce::jlimit (0.0f, 1.0f,
                (db - kMinDecibels) / (kMaxDecibels - kMinDecibels));
        }
    }

    /** Per-bin level in 0..1, exactly what getByteFrequencyData()/255 gives. */
    const std::array<float, kBins>& frequency() const noexcept { return level; }
    /** The raw window, in -1..1. getByteTimeDomainData() without the +128. */
    const std::array<float, kFftSize>& waveform() const noexcept { return time; }

private:
    juce::dsp::FFT fft;
    std::array<float, kFftSize> window {};
    std::array<float, kFftSize> time {};
    std::array<float, kFftSize * 2> scratch {};
    std::array<float, kBins> smoothed {};
    std::array<float, kBins> level {};
};

}  // namespace atlasflow

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <juce_graphics/juce_graphics.h>

/*  A path drawn across the atlas, and the walk along it.

    This is the gesture the paper is about: a stroke across the map becomes
    timbre automation for the take. A port of the trajectory half of
    atlas-flow-web-demo/app.js, including its speed profile, which is the part
    that makes the walk feel played rather than scanned.

    The profile is curvature-based: tight turns are taken slowly, the result is
    normalised against the *median* rather than the mean so one scribbled
    hairpin colours the walk instead of halting it, both ends are eased, and the
    whole thing is scaled by its harmonic mean so the lap time the user asked
    for is still the lap time they get.

    `PathData` is immutable once built. The editor builds one and hands it over;
    the audio thread copies it into its own buffer and then reads lock-free.
*/

namespace atlasflow {

struct PathData
{
    static constexpr int kSamples = 256;
    static constexpr float kEaseSpan = 0.12f;

    std::vector<juce::Point<float>> points;   // normalised 0..1 map coordinates
    std::vector<float> lengths;               // cumulative arc length
    std::vector<float> speed;                 // kSamples entries, harmonic mean 1
    float total = 0.0f;

    bool valid() const noexcept { return points.size() >= 2 && total > 1.0e-6f; }

    void build (std::vector<juce::Point<float>> stroke)
    {
        points = std::move (stroke);
        lengths.assign (points.size(), 0.0f);
        total = 0.0f;
        for (size_t i = 1; i < points.size(); ++i)
        {
            total += points[i].getDistanceFrom (points[i - 1]);
            lengths[i] = total;
        }
        if (! valid())
        {
            speed.clear();
            return;
        }
        buildSpeedProfile();
    }

    juce::Point<float> at (float phase) const noexcept
    {
        if (points.size() < 2)
            return points.empty() ? juce::Point<float> {} : points.front();

        const float wanted = std::clamp (phase, 0.0f, 1.0f) * total;
        size_t index = 1;
        while (index < lengths.size() - 1 && lengths[index] < wanted)
            ++index;
        const float before = lengths[index - 1];
        const float span = std::max (1.0e-9f, lengths[index] - before);
        const float ratio = std::clamp ((wanted - before) / span, 0.0f, 1.0f);
        return points[index - 1] + (points[index] - points[index - 1]) * ratio;
    }

    float speedAt (float phase) const noexcept
    {
        if (speed.empty())
            return 1.0f;
        const float place = std::clamp (phase, 0.0f, 1.0f) * (float) (speed.size() - 1);
        const size_t index = (size_t) place;
        const size_t next = std::min (index + 1, speed.size() - 1);
        return speed[index] + (speed[next] - speed[index]) * (place - (float) index);
    }

private:
    void buildSpeedProfile()
    {
        const int count = kSamples;
        std::vector<juce::Point<float>> samples ((size_t) count);
        for (int i = 0; i < count; ++i)
            samples[(size_t) i] = at ((float) i / (float) (count - 1));

        const float ds = std::max (total / (float) (count - 1), 1.0e-6f);
        speed.assign ((size_t) count, 1.0f);
        for (int i = 1; i < count - 1; ++i)
        {
            const auto d = samples[(size_t) i - 1] - samples[(size_t) i] * 2.0f + samples[(size_t) i + 1];
            const float curvature = std::hypot (d.x, d.y) / (ds * ds);
            speed[(size_t) i] = std::cbrt (1.0f / std::max (curvature, 1.0e-6f));
        }
        speed[0] = speed[1];
        speed[(size_t) count - 1] = speed[(size_t) count - 2];

        auto sorted = speed;
        std::nth_element (sorted.begin(), sorted.begin() + count / 2, sorted.end());
        const float median = std::max (sorted[(size_t) count / 2], 1.0e-6f);
        for (auto& v : speed)
            v = std::clamp (v / median, 0.4f, 1.7f);

        for (int i = 0; i < count; ++i)
        {
            const float phase = (float) i / (float) (count - 1);
            const float edge = std::min (phase, 1.0f - phase) / kEaseSpan;
            if (edge < 1.0f)
                speed[(size_t) i] *= 0.3f + 0.7f * (edge * edge * (3.0f - 2.0f * edge));
        }

        // Harmonic mean, because the walk spends time inversely to speed: this
        // is what keeps "10 seconds one way" true whatever the stroke's shape.
        double harmonic = 0.0;
        for (float v : speed)
            harmonic += 1.0 / std::max (v, 1.0e-6f);
        harmonic /= (double) count;
        for (auto& v : speed)
            v *= (float) harmonic;
    }
};

/** The moving head: where along the path the timbre currently is. */
class PathWalker
{
public:
    void reset() noexcept { phase = 0.0f; forward = true; }

    /** Advance by `samples` and return the current map position.
        There and back, so a stroke reads as a sweep rather than a jump cut at
        the end of every lap. */
    juce::Point<float> advance (const PathData& path, int samples,
                                double sampleRate, float lapSeconds) noexcept
    {
        const float elapsed = (float) ((double) samples / std::max (sampleRate, 1.0));
        const float lap = std::max (0.5f, lapSeconds);
        phase += (forward ? 1.0f : -1.0f) * elapsed / lap * path.speedAt (phase);
        if (phase >= 1.0f) { phase = 1.0f; forward = false; }
        if (phase <= 0.0f) { phase = 0.0f; forward = true; }
        return path.at (phase);
    }

    float position() const noexcept { return phase; }

private:
    float phase = 0.0f;
    bool forward = true;
};

}  // namespace atlasflow

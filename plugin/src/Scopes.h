#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "Analyser.h"
#include "Palette.h"
#include "Strings.h"

namespace atlasflow {

/*  Five views of one analyser pass, ported from the desktop instrument.

    Each view keeps its own history because they scroll at different rates and
    would otherwise fight over one buffer. One analyser pass per frame feeds
    all of them, so the cost is the drawing, not the analysis.

    Measured across the atlas: centroids run 90-1033 Hz and 85% rolloff never
    passes 1.4 kHz, so a pad puts nearly everything under ~2 kHz. Drawing up to
    Nyquist spent three quarters of the pane on silence and squeezed every real
    difference into the left edge, which is why the waterfall looked static.
*/

inline constexpr float kScopeCeiling = 4000.0f;

inline float scopeTop (double rate)
{
    return juce::jmin (kScopeCeiling, (float) rate * 0.5f);
}

inline float toMel (float frequency) { return 2595.0f * std::log10 (1.0f + frequency / 700.0f); }
inline float fromMel (float value) { return 700.0f * (std::pow (10.0f, value / 2595.0f) - 1.0f); }

/** Which views exist, in the order the menu offers them. */
enum class ScopeKind { spectrogram = 0, ridge = 1, loudness = 2, spectrum = 3, wave = 4 };
inline constexpr int kScopeKindCount = 5;

inline const char* scopeName (ScopeKind kind)
{
    switch (kind)
    {
        case ScopeKind::spectrogram: return "SPECTROGRAM";
        case ScopeKind::ridge:       return "WATERFALL";
        case ScopeKind::loudness:    return "LOUDNESS";
        case ScopeKind::spectrum:    return "SPECTRUM";
        case ScopeKind::wave:        return "WAVE";
    }
    return "SPECTROGRAM";
}

/** One pane. Owns its own history and nothing else. */
class ScopeView
{
public:
    void setKind (ScopeKind next)
    {
        if (kind == next)
            return;
        kind = next;
        reset();
    }

    ScopeKind getKind() const noexcept { return kind; }

    /** Throw away every history, which is what a resize or a palette change
        means: the strip was painted in the old size and the old colours, so it
        has to go rather than be drawn over. */
    void reset()
    {
        strip = {};
        ridges.clear();
        levels.clear();
        peaks.clear();
        rows.clear();
        carry = 0.0f;
    }

    void paint (juce::Graphics& g, juce::Rectangle<int> bounds, const Analyser& analyser,
                const Palette& theme, float elapsed, double rate, bool dropped, int language = kEnglish)
    {
        if (bounds.getWidth() < 8 || bounds.getHeight() < 8)
            return;

        if (width != bounds.getWidth() || height != bounds.getHeight())
        {
            width = bounds.getWidth();
            height = bounds.getHeight();
            reset();
        }

        g.setColour (theme.background);
        g.fillRect (bounds);

        juce::Graphics::ScopedSaveState saved (g);
        g.reduceClipRegion (bounds);
        g.addTransform (juce::AffineTransform::translation ((float) bounds.getX(), (float) bounds.getY()));

        switch (kind)
        {
            case ScopeKind::spectrogram: drawSpectrogram (g, analyser, theme, elapsed, rate, dropped); break;
            case ScopeKind::ridge:       drawRidge (g, analyser, theme, elapsed, rate); break;
            case ScopeKind::loudness:    drawLoudness (g, analyser, theme, elapsed, language); break;
            case ScopeKind::spectrum:    drawSpectrum (g, analyser, theme, elapsed, rate); break;
            case ScopeKind::wave:        drawWave (g, analyser, theme); break;
        }
    }

private:
    // --- spectrogram -------------------------------------------------------

    /** Mel-spaced bin ranges, one per output row.

        Mel rather than log or linear because it spends pixels where a pad's
        energy actually is. Recomputed only when the pane or the rate changes.
    */
    void buildRows (double rate)
    {
        rows.resize ((size_t) height);
        const float top = toMel (scopeTop (rate));
        const float bottom = toMel (40.0f);
        const float nyquist = (float) rate * 0.5f;
        for (int row = 0; row < height; ++row)
        {
            const float high = fromMel (bottom + (top - bottom) * (1.0f - (float) row / (float) height));
            const float low  = fromMel (bottom + (top - bottom) * (1.0f - (float) (row + 1) / (float) height));
            const int from = juce::jmax (0, (int) std::floor (low / nyquist * (float) Analyser::kBins));
            const int to = juce::jlimit (from + 1, Analyser::kBins,
                                         (int) std::ceil (high / nyquist * (float) Analyser::kBins));
            rows[(size_t) row] = { from, to };
        }
    }

    void drawSpectrogram (juce::Graphics& g, const Analyser& analyser, const Palette& theme,
                          float elapsed, double rate, bool dropped)
    {
        if (! strip.isValid())
        {
            strip = juce::Image (juce::Image::ARGB, width, height, true);
            juce::Graphics clear (strip);
            clear.setColour (theme.background);
            clear.fillAll();
        }
        if (rows.size() != (size_t) height)
            buildRows (rate);

        carry += elapsed * 64.0f;
        const int columns = juce::jlimit (0, 10, (int) std::floor (carry));
        carry -= (float) columns;

        if (columns > 0)
        {
            scrollLeft (columns);
            const auto& level = analyser.frequency();
            const juce::Image::BitmapData pixels (strip, juce::Image::BitmapData::readWrite);
            for (int row = 0; row < height; ++row)
            {
                const auto [low, high] = rows[(size_t) row];
                float sum = 0.0f;
                for (int bin = low; bin < high; ++bin)
                    sum += level[(size_t) bin];
                const float value = std::pow (sum / (float) juce::jmax (1, high - low), 1.2f);
                const auto colour = theme.rampAt (value);
                for (int column = width - columns; column < width; ++column)
                    pixels.setPixelColour (column, row, colour);
            }
            if (dropped)
            {
                // A dropout is a fact about the audio, not about the model, and
                // it belongs on the picture of the audio.
                for (int row = juce::jmax (0, height - 3); row < height; ++row)
                    for (int column = width - columns; column < width; ++column)
                        pixels.setPixelColour (column, row, theme.warn);
            }
        }

        g.drawImageAt (strip, 0, 0);
        melGrid (g, theme, rate);
    }

    /** Shift the strip `columns` pixels left, in place.

        A canvas can blit itself with globalCompositeOperation = 'copy'; an
        Image cannot be drawn onto itself, so the rows move by memmove instead.
        Same result, and it does not allocate a second image per frame.
    */
    void scrollLeft (int columns)
    {
        const juce::Image::BitmapData pixels (strip, juce::Image::BitmapData::readWrite);
        const int shift = juce::jlimit (0, width, columns);
        const auto bytes = (size_t) pixels.pixelStride;
        for (int row = 0; row < height; ++row)
        {
            auto* line = pixels.getLinePointer (row);
            std::memmove (line, line + (size_t) shift * bytes, (size_t) (width - shift) * bytes);
        }
    }

    void melGrid (juce::Graphics& g, const Palette& theme, double rate) const
    {
        const float top = toMel (scopeTop (rate));
        const float bottom = toMel (40.0f);
        g.setFont (juce::FontOptions (9.0f));
        for (const float frequency : { 100.0f, 250.0f, 500.0f, 1000.0f, 2000.0f })
        {
            if (frequency >= scopeTop (rate))
                continue;
            const float y = (float) height * (1.0f - (toMel (frequency) - bottom) / (top - bottom));
            g.setColour (theme.tint (0.10f));
            g.drawLine (0.0f, y, (float) width, y, 1.0f);
            g.setColour (theme.tint (0.34f));
            g.drawText (label (frequency), 4, (int) y - 12, 40, 10, juce::Justification::left, false);
        }
    }

    // --- waterfall ---------------------------------------------------------

    void drawRidge (juce::Graphics& g, const Analyser& analyser, const Palette& theme,
                    float elapsed, double rate)
    {
        constexpr int bands = 72;
        const auto line = logBands (analyser, rate, bands);

        carry += elapsed * 22.0f;
        while (carry >= 1.0f)
        {
            carry -= 1.0f;
            ridges.push_front (line);
            if (ridges.size() > 34)
                ridges.pop_back();
        }

        const float step = (float) height / 38.0f;
        const float amplitude = (float) height / 3.4f;
        juce::Graphics::ScopedSaveState saved (g);
        g.reduceClipRegion (juce::Rectangle<int> { 0, 0, width, height - 10 });

        // Back to front, each ridge filled with the background so it occludes
        // the ones behind it. That occlusion is the whole effect.
        for (int index = (int) ridges.size() - 1; index >= 0; --index)
        {
            const auto& ridge = ridges[(size_t) index];
            const float base = (float) height - 8.0f - (float) index * step;
            const float fade = 1.0f - (float) index / (float) ridges.size();

            juce::Path shape;
            shape.startNewSubPath (0.0f, base);
            for (int band = 0; band < (int) ridge.size(); ++band)
            {
                const float x = (float) band / (float) (ridge.size() - 1) * (float) width;
                shape.lineTo (x, base - std::pow (ridge[(size_t) band], 1.3f) * amplitude);
            }
            shape.lineTo ((float) width, base);
            shape.closeSubPath();

            g.setColour (theme.background);
            g.fillPath (shape);
            g.setColour (theme.rampAt (0.15f + 0.85f * fade));
            g.strokePath (shape, juce::PathStrokeType (1.0f));
        }
        logGrid (g, theme, rate, true);
    }

    // --- spectrum ----------------------------------------------------------

    void drawSpectrum (juce::Graphics& g, const Analyser& analyser, const Palette& theme,
                       float elapsed, double rate)
    {
        const int bars = juce::jmin (180, juce::jmax (1, width / 3));
        if ((int) peaks.size() != bars)
            peaks.assign ((size_t) bars, 0.0f);

        const auto line = logBands (analyser, rate, bars);
        for (int index = 0; index < bars; ++index)
        {
            const float value = line[(size_t) index];
            peaks[(size_t) index] = juce::jmax (value, peaks[(size_t) index] - elapsed * 0.55f);
            const float x = (float) index / (float) bars * (float) width;
            const float barWidth = (float) width / (float) bars - 1.0f;
            g.setColour (theme.rampAt (0.2f + 0.8f * value));
            g.fillRect (x, (float) height * (1.0f - value), barWidth, (float) height * value);
            g.setColour (theme.tint (0.85f));
            g.fillRect (x, (float) height * (1.0f - peaks[(size_t) index]) - 1.0f, barWidth, 1.0f);
        }
        logGrid (g, theme, rate, false);
    }

    // --- wave --------------------------------------------------------------

    /** Triggered on a rising zero crossing, so the wave stands still. */
    void drawWave (juce::Graphics& g, const Analyser& analyser, const Palette& theme)
    {
        const auto& samples = analyser.waveform();
        const int count = (int) samples.size() / 2;
        int trigger = 0;
        for (int index = 1; index < count; ++index)
            if (samples[(size_t) (index - 1)] < 0.0f && samples[(size_t) index] >= 0.0f)
            {
                trigger = index;
                break;
            }

        g.setColour (theme.tint (0.14f));
        g.drawLine (0.0f, (float) height * 0.5f, (float) width, (float) height * 0.5f, 1.0f);

        juce::Path trace;
        for (int index = 0; index < count; ++index)
        {
            const float value = samples[(size_t) juce::jmin ((int) samples.size() - 1, trigger + index)];
            const float x = (float) index / (float) (count - 1) * (float) width;
            const float y = (float) height * 0.5f - value * (float) height * 0.42f;
            index == 0 ? trace.startNewSubPath (x, y) : trace.lineTo (x, y);
        }
        g.setColour (theme.accent2);
        g.strokePath (trace, juce::PathStrokeType (1.2f));
    }

    // --- loudness ----------------------------------------------------------

    /** RMS and peak over time.

        The desktop version draws a third, dashed trace: the multiband
        compressor's gain reduction. There is no compressor in the plugin --
        the host's own chain is where that belongs, and a plugin that quietly
        levels its own output is lying about the model's loudness, which is one
        of the gates the model does not pass. So two traces, not three, and the
        pane says what it is showing.
    */
    void drawLoudness (juce::Graphics& g, const Analyser& analyser, const Palette& theme,
                       float elapsed, int language)
    {
        const auto& samples = analyser.waveform();
        float sum = 0.0f, peak = 0.0f;
        for (const float value : samples)
        {
            sum += value * value;
            peak = juce::jmax (peak, std::abs (value));
        }
        const float rms = std::sqrt (sum / (float) samples.size());

        carry += elapsed * 40.0f;
        while (carry >= 1.0f)
        {
            carry -= 1.0f;
            levels.push_back ({ juce::Decibels::gainToDecibels (rms, -100.0f),
                                juce::Decibels::gainToDecibels (peak, -100.0f) });
            while ((int) levels.size() > width)
                levels.pop_front();
        }

        // -72 dB of range put every trace in the top third. The model's output
        // sits around -26 RMS and -17 peak, so this window is the part that
        // carries signal.
        constexpr float floorDb = -42.0f;
        auto toY = [this] (float db)
        {
            return (float) height * (1.0f - juce::jlimit (0.0f, 1.0f, (db - floorDb) / -floorDb));
        };

        g.setFont (juce::FontOptions (9.0f));
        for (const float db : { -6.0f, -12.0f, -24.0f, -36.0f })
        {
            const float y = toY (db);
            g.setColour (theme.tint (0.09f));
            g.drawLine (0.0f, y, (float) width, y, 1.0f);
            g.setColour (theme.tint (0.32f));
            g.drawText (juce::String ((int) db), 4, (int) y - 12, 40, 10, juce::Justification::left, false);
        }

        auto trace = [&] (bool wantPeak, juce::Colour colour, float thickness)
        {
            juce::Path line;
            for (size_t index = 0; index < levels.size(); ++index)
            {
                const float x = (float) width - (float) levels.size() + (float) index;
                const float y = toY (wantPeak ? levels[index].peak : levels[index].rms);
                index == 0 ? line.startNewSubPath (x, y) : line.lineTo (x, y);
            }
            g.setColour (colour);
            g.strokePath (line, juce::PathStrokeType (thickness));
        };
        trace (true, theme.accent.withAlpha (0.55f), 1.0f);
        trace (false, theme.ink, 1.4f);

        if (! levels.empty())
        {
            const auto latest = levels.back();
            g.setColour (theme.ink);
            g.drawText ("RMS " + juce::String (latest.rms, 1) + "  "
                            + tr (text::peak, language) + " " + juce::String (latest.peak, 1) + " dB",
                        0, 2, width - 6, 10, juce::Justification::right, false);
        }
    }

    // --- shared ------------------------------------------------------------

    /** Peak per band on a log frequency axis, which both the waterfall and the
        spectrum want and neither should compute twice. */
    std::vector<float> logBands (const Analyser& analyser, double rate, int bands) const
    {
        const auto& level = analyser.frequency();
        const float nyquist = (float) rate * 0.5f;
        constexpr float lowest = 40.0f;
        const float span = std::log2 (scopeTop (rate) / lowest);

        std::vector<float> out ((size_t) bands, 0.0f);
        for (int index = 0; index < bands; ++index)
        {
            const float from = lowest * std::pow (2.0f, span * (float) index / (float) bands);
            const float to = lowest * std::pow (2.0f, span * (float) (index + 1) / (float) bands);
            const int start = juce::jmax (0, (int) std::floor (from / nyquist * (float) Analyser::kBins));
            const int end = juce::jlimit (start + 1, Analyser::kBins,
                                          (int) std::ceil (to / nyquist * (float) Analyser::kBins));
            float peak = 0.0f;
            for (int bin = start; bin < end; ++bin)
                peak = juce::jmax (peak, level[(size_t) bin]);
            out[(size_t) index] = peak;
        }
        return out;
    }

    void logGrid (juce::Graphics& g, const Palette& theme, double rate, bool faint) const
    {
        constexpr float lowest = 40.0f;
        const float span = std::log2 (scopeTop (rate) / lowest);
        g.setFont (juce::FontOptions (9.0f));
        for (const float frequency : { 100.0f, 250.0f, 500.0f, 1000.0f, 2000.0f })
        {
            if (frequency >= scopeTop (rate))
                continue;
            const float x = std::log2 (frequency / lowest) / span * (float) width;
            g.setColour (theme.tint (faint ? 0.05f : 0.09f));
            g.drawLine (x, 0.0f, x, (float) height - 10.0f, 1.0f);
            g.setColour (theme.tint (0.34f));
            g.drawText (label (frequency), (int) x - 20, height - 12, 40, 10, juce::Justification::centred, false);
        }
    }

    static juce::String label (float frequency)
    {
        return frequency >= 1000.0f ? juce::String (frequency / 1000.0f, 0) + "k"
                                    : juce::String ((int) frequency);
    }

    struct Level { float rms, peak; };

    ScopeKind kind = ScopeKind::spectrogram;
    int width = 0, height = 0;
    float carry = 0.0f;
    juce::Image strip;
    std::deque<std::vector<float>> ridges;
    std::deque<Level> levels;
    std::vector<float> peaks;
    std::vector<std::pair<int, int>> rows;
};

}  // namespace atlasflow

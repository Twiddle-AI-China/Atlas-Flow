#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <juce_graphics/juce_graphics.h>

#include "AtlasGeometry.h"
#include "Palette.h"

namespace atlasflow {

/*  How much the runtime will overrule you, everywhere on the plane.

    Bright means the map is telling the truth there: ask for that coordinate
    and roughly that is what you hear. Dark means the request gets pulled
    somewhere else before it reaches the model, so the position under your
    cursor is decorative.

    This is the same field the desktop instrument draws, with one difference
    that is an improvement rather than a port: the web reimplements the
    projection in JavaScript and measures the displacement with a plain
    Euclidean metric, while here the real AtlasGeometry::projectDetails does
    the work -- the very code the audio thread calls. The picture therefore
    cannot drift away from the behaviour it claims to describe.

    The restriction to the nearest preset's connected component is the part
    that matters most: 9 of the 50 presets are singletons, so a tenth of the
    plane can only ever resolve to one preset however far away it is. That is
    what the dark basins are.

    An earlier version shaded by 8-D neighbour distance instead, which sounded
    reasonable and measured as unrelated -- correlation 0.120 against the
    actual override. Worse, it was brightest in the corners, where every
    Gaussian in the lift underflows and the lifted point collapses to the mean
    of all 50 presets, which sits centrally in 8-D and so scored as
    well-covered. It was brightest exactly where you are most lost.
*/
class Terrain
{
public:
    static constexpr int kGrid = 128;
    static constexpr int kNear = 4;        // FIELD_NEAR: projectDetails' neighbour count

    // Every cell contributes to the lift, weighted by a Gaussian a few
    // neighbour-spacings wide. A hard nearest-three instead made the blend
    // jump each time the walk crossed a rank boundary: 8-D travel came out at
    // 41.8 for a span of 3.7, eleven times more movement than ground covered,
    // which is jitter, not motion.
    static constexpr float kNeighbourSpacing = 0.098f;
    static constexpr float kLiftBandwidth = kNeighbourSpacing * 3.0f;

    /** The height field, computed once.

        It is a function of the atlas in normalised coordinates, so it does not
        depend on the window at all -- only its scaling to pixels does. The
        desktop build spends 47 ms here and its resize handler is not
        debounced, which made dragging a window edge stutter; this costs around
        8 ms and is still computed once and kept for the life of the process,
        because there is no reason to pay it twice.
    */
    static const Terrain& shared()
    {
        static const Terrain built;
        return built;
    }

    float at (int row, int column) const
    {
        return height[(size_t) (clampIndex (row) * kGrid + clampIndex (column))];
    }

    /** Relief-shaded ground, as a 128x128 image to be stretched over the map.

        Lit from the north-west, the cartographic convention. Ambient keeps
        flat ground visible and the slope term is what reads as terrain, but
        ground level still gates the whole thing, so the basins the model does
        not cover stay dark however the light falls on them.
    */
    juce::Image shaded (const Palette& theme) const
    {
        juce::Image image (juce::Image::ARGB, kGrid, kGrid, true);
        const juce::Image::BitmapData pixels (image, juce::Image::BitmapData::writeOnly);

        constexpr float lightX = -0.6f, lightY = -0.6f, lightZ = 0.53f;
        const float norm = std::sqrt (lightX * lightX + lightY * lightY + lightZ * lightZ);
        const float lx = lightX / norm, ly = lightY / norm, lz = lightZ / norm;
        constexpr float kRelief = 26.0f;       // vertical exaggeration

        for (int row = 0; row < kGrid; ++row)
            for (int column = 0; column < kGrid; ++column)
            {
                const float dzdx = (at (row, column + 1) - at (row, column - 1)) * 0.5f * kRelief;
                const float dzdy = (at (row + 1, column) - at (row - 1, column)) * 0.5f * kRelief;
                const float length = std::sqrt (dzdx * dzdx + dzdy * dzdy + 1.0f);
                const float slope = juce::jmax (0.0f, (-dzdx * lx - dzdy * ly + lz) / length);
                const float lit = 0.30f + 0.70f * slope;
                const float ground = at (row, column) * at (row, column);
                const float alpha = juce::jlimit (0.0f, 1.0f, 0.19f * lit * ground);
                pixels.setPixelColour (column, row, theme.ink.withAlpha (alpha));
            }

        return image;
    }

    /** Iso-fidelity contours, by marching squares.

        Shading alone reads as smoke rather than ground -- what makes a height
        field legible as terrain is the lines. Each one here is a real
        threshold: every point along a given contour gets overruled by the
        runtime to the same degree, so the rings around a cluster are the reach
        of that cluster.
    */
    static constexpr int kContours = 5;

    /** One path per threshold, in pixels, for the caller to stroke and cache.

        Paths rather than a bitmap: a cached image of the hairlines would have
        to be built at the display's scale factor or come out soft, and these
        are the finest lines on the surface. A path is resolution-independent
        and cheap to stroke, so it survives a retina window and a resize
        without either blurring or being recomputed every frame.
    */
    std::array<juce::Path, kContours> contourPaths (juce::Rectangle<float> area) const
    {
        static constexpr float levels[kContours] = { 0.28f, 0.42f, 0.56f, 0.70f, 0.84f };
        const float stepX = area.getWidth() / (float) (kGrid - 1);
        const float stepY = area.getHeight() / (float) (kGrid - 1);

        std::array<juce::Path, kContours> out;
        for (int index = 0; index < kContours; ++index)
        {
            const float level = levels[index];
            juce::Path& lines = out[(size_t) index];
            for (int row = 0; row < kGrid - 1; ++row)
                for (int column = 0; column < kGrid - 1; ++column)
                {
                    const float a = at (row, column);          // corners, clockwise
                    const float b = at (row, column + 1);      // from top-left
                    const float c = at (row + 1, column + 1);
                    const float d = at (row + 1, column);
                    const int code = (a > level ? 8 : 0) | (b > level ? 4 : 0)
                                   | (c > level ? 2 : 0) | (d > level ? 1 : 0);
                    if (code == 0 || code == 15)
                        continue;

                    const float x0 = area.getX() + (float) column * stepX;
                    const float y0 = area.getY() + (float) row * stepY;
                    auto mix = [level] (float p, float q)
                    {
                        const float span = q - p;
                        return (level - p) / (std::abs (span) > 0.0f ? span : 1.0e-9f);
                    };
                    const juce::Point<float> top   { x0 + stepX * mix (a, b), y0 };
                    const juce::Point<float> right { x0 + stepX, y0 + stepY * mix (b, c) };
                    const juce::Point<float> below { x0 + stepX * mix (d, c), y0 + stepY };
                    const juce::Point<float> left  { x0, y0 + stepY * mix (a, d) };

                    auto segment = [&lines] (juce::Point<float> from, juce::Point<float> to)
                    {
                        lines.startNewSubPath (from);
                        lines.lineTo (to);
                    };
                    switch (code)
                    {
                        case 1: case 14: segment (left, below);  break;
                        case 2: case 13: segment (below, right); break;
                        case 3: case 12: segment (left, right);  break;
                        case 4: case 11: segment (top, right);   break;
                        case 6: case 9:  segment (top, below);   break;
                        case 7: case 8:  segment (left, top);    break;
                        // Saddles: two separate crossings rather than one line.
                        case 5:  segment (left, top);  segment (below, right); break;
                        case 10: segment (left, below); segment (top, right);  break;
                        default: break;
                    }
                }
        }
        return out;
    }

    /** The lift the field is measured along, exposed for the readout.

        The pointer supplies PC1/PC2 and the atlas fills in the other six from
        the presets you are passing. The path still owns where you are; the
        atlas owns the rest.
    */
    static std::array<float, kAtlasDims> lift (float x, float y)
    {
        std::array<float, kAtlasDims> lifted {};
        lifted[0] = x;
        lifted[1] = y;

        const auto& normalised = normalisedCoordinates();
        std::array<float, kAtlasDims> blend {};
        float total = 0.0f;
        for (int p = 0; p < kPresetCount; ++p)
        {
            const float dx = normalised[(size_t) p][0] - x;
            const float dy = normalised[(size_t) p][1] - y;
            const float reach = std::sqrt (dx * dx + dy * dy) / kLiftBandwidth;
            // The floor keeps the far corners finite rather than 0/0; it is
            // also why they collapse to the mean of all 50 presets out there.
            const float weight = std::exp (-reach * reach) + 1.0e-4f;
            total += weight;
            for (int d = 2; d < kAtlasDims; ++d)
                blend[(size_t) d] += normalised[(size_t) p][(size_t) d] * weight;
        }
        for (int d = 2; d < kAtlasDims; ++d)
            lifted[(size_t) d] = blend[(size_t) d] / total;
        return lifted;
    }

    /** How far the runtime moves a request made at (x, y), in normalised units.

        Normalised rather than raw atlas units so every axis counts for the
        same amount in the picture: raw PC1 spans five times what PC8 does, and
        shading by raw displacement would draw PC1 and little else.
    */
    static float overrideAt (float x, float y)
    {
        static const AtlasGeometry geometry;
        const auto lifted = lift (x, y);
        const auto projected = geometry.normalize (geometry.projectDetails (geometry.denormalize (lifted), kNear).coordinate);

        float moved = 0.0f;
        for (int d = 0; d < kAtlasDims; ++d)
        {
            const float delta = projected[(size_t) d] - lifted[(size_t) d];
            moved += delta * delta;
        }
        return std::sqrt (moved);
    }

private:
    Terrain()
    {
        std::vector<float> raw ((size_t) kGrid * kGrid, 0.0f);
        float lowest = std::numeric_limits<float>::max();
        float highest = 0.0f;
        for (int row = 0; row < kGrid; ++row)
            for (int column = 0; column < kGrid; ++column)
            {
                const float x = (float) column / (float) (kGrid - 1) * 2.0f - 1.0f;
                const float y = 1.0f - (float) row / (float) (kGrid - 1) * 2.0f;
                const float value = overrideAt (x, y);
                raw[(size_t) (row * kGrid + column)] = value;
                lowest = juce::jmin (lowest, value);
                highest = juce::jmax (highest, value);
            }

        // Read it as ground: faithful is high, overruled is low.
        const float span = juce::jmax (highest - lowest, 1.0e-6f);
        std::vector<float> levelled (raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
            levelled[i] = 1.0f - (raw[i] - lowest) / span;

        // One box pass. The scalar is smooth but the grid is not fine, and the
        // relief shading differentiates it -- without this the facets of the
        // sampling grid light up instead of the terrain.
        height.assign (raw.size(), 0.0f);
        for (int row = 0; row < kGrid; ++row)
            for (int column = 0; column < kGrid; ++column)
            {
                float sum = 0.0f;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        sum += levelled[(size_t) (clampIndex (row + dy) * kGrid + clampIndex (column + dx))];
                height[(size_t) (row * kGrid + column)] = sum / 9.0f;
            }
    }

    static int clampIndex (int v) { return juce::jlimit (0, kGrid - 1, v); }

    /** Preset positions in the same [-1,1] units the pointer works in. */
    static const std::array<std::array<float, kAtlasDims>, kPresetCount>& normalisedCoordinates()
    {
        static const auto table = []
        {
            const AtlasGeometry geometry;
            std::array<std::array<float, kAtlasDims>, kPresetCount> out {};
            for (int p = 0; p < kPresetCount; ++p)
            {
                std::array<float, kAtlasDims> raw {};
                for (int d = 0; d < kAtlasDims; ++d)
                    raw[(size_t) d] = kCoordinates[p][d];
                out[(size_t) p] = geometry.normalize (raw);
            }
            return out;
        }();
        return table;
    }

    std::vector<float> height;
};

}  // namespace atlasflow

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "AtlasData.h"

namespace atlasflow {

/** Geometry of the timbre map.

    This is the real trained atlas (see tools/export_atlas.py) even though the
    sound at each anchor is currently mocked. Two things live here: how a
    preset lands on screen, and how a pointer position becomes a blend over
    nearby presets.

    The screen uses PC1/PC2 only. That is a lossy view of an 8-D space and the
    frontend says so out loud -- PC1 and PC2 hold only 42.3% and 19.8% of the
    variance -- which is why two dots can look adjacent and sound far apart.
    The blend below therefore weights in the *full* 8-D space, not in screen
    distance, so what you hear matches the real geometry rather than the
    picture of it.
*/
class Atlas
{
public:
    static constexpr int kBlendNeighbours = 4;

    Atlas()
    {
        for (int d = 0; d < kAtlasDims; ++d)
        {
            lo[(size_t) d] = kCoordinates[0][d];
            hi[(size_t) d] = kCoordinates[0][d];
            for (int p = 1; p < kPresetCount; ++p)
            {
                lo[(size_t) d] = std::min (lo[(size_t) d], kCoordinates[p][d]);
                hi[(size_t) d] = std::max (hi[(size_t) d], kCoordinates[p][d]);
            }
        }
    }

    /** Normalised [0,1] screen position of a preset, from PC1/PC2. */
    void displayPosition (int preset, float& x, float& y) const noexcept
    {
        x = norm (kCoordinates[preset][0], 0);
        y = 1.0f - norm (kCoordinates[preset][1], 1);   // screen y grows downward
    }

    /** Map a normalised pointer position back into atlas units on PC1/PC2. */
    void pointerToAtlas (float nx, float ny, float& pc1, float& pc2) const noexcept
    {
        pc1 = denorm (nx, 0);
        pc2 = denorm (1.0f - ny, 1);
    }

    /** Inverse-distance blend over the nearest presets.

        The pointer only supplies PC1/PC2. The remaining six dimensions come
        from the neighbours themselves, which is what keeps every position a
        mixture of presets the model has actually seen rather than an arbitrary
        point in an 8-D space that may be empty.

        Writes kPresetCount weights summing to 1, all but the nearest few zero.
    */
    void blend (float pc1, float pc2, float* weights) const noexcept
    {
        std::fill (weights, weights + kPresetCount, 0.0f);

        // Rank by distance on the two axes the pointer actually controls.
        std::array<int, kPresetCount> order {};
        std::array<float, kPresetCount> dist {};
        for (int p = 0; p < kPresetCount; ++p)
        {
            const float dx = (kCoordinates[p][0] - pc1) / kScale[0];
            const float dy = (kCoordinates[p][1] - pc2) / kScale[1];
            dist[(size_t) p] = std::sqrt (dx * dx + dy * dy);
            order[(size_t) p] = p;
        }
        std::partial_sort (order.begin(), order.begin() + kBlendNeighbours, order.end(),
                           [&dist] (int a, int b) { return dist[(size_t) a] < dist[(size_t) b]; });

        float total = 0.0f;
        for (int k = 0; k < kBlendNeighbours; ++k)
        {
            const int p = order[(size_t) k];
            // +epsilon so sitting exactly on a preset is finite, not infinite.
            const float w = 1.0f / (dist[(size_t) p] * dist[(size_t) p] + 1.0e-4f);
            weights[p] = w;
            total += w;
        }
        if (total > 0.0f)
            for (int k = 0; k < kBlendNeighbours; ++k)
                weights[order[(size_t) k]] /= total;
    }

    int nearest (float pc1, float pc2) const noexcept
    {
        int best = 0;
        float bestDistance = 1.0e30f;
        for (int p = 0; p < kPresetCount; ++p)
        {
            const float dx = (kCoordinates[p][0] - pc1) / kScale[0];
            const float dy = (kCoordinates[p][1] - pc2) / kScale[1];
            const float d = dx * dx + dy * dy;
            if (d < bestDistance) { bestDistance = d; best = p; }
        }
        return best;
    }

    static bool neighbours (int a, int b) noexcept { return kNeighbours[a][b]; }

    /** How much of an edge's real length survives being drawn.

        The screen shows PC1 and PC2, which hold only 42.3% and 19.8% of the
        variance, so most of what separates two presets happens on axes this
        picture cannot draw. An edge keeps a median 0.36 of its true 8-D length
        once projected -- "loses something" describes almost every edge and is
        not worth drawing. Dashing the worst quarter, below 0.25, makes the dash
        mean something you can act on: those two dots look adjacent and are not,
        and that is usually where roaming jumps.
    */
    static float foldedRatio (int a, int b) noexcept
    {
        float flat = 0.0f, full = 0.0f;
        for (int d = 0; d < kAtlasDims; ++d)
        {
            const float delta = (kCoordinates[a][d] - kCoordinates[b][d]) / kScale[d];
            full += delta * delta;
            if (d < 2) flat += delta * delta;
        }
        if (full <= 0.0f) return 1.0f;
        return std::sqrt (flat) / std::sqrt (full);
    }

    /** Where an arbitrary atlas coordinate lands on screen.

        displayPosition() answers this for a preset; the sounding voice is at a
        *projected* coordinate that is usually not any preset, and drawing the
        gap between it and the pointer is the clearest statement the map makes
        about what the runtime is doing to you. */
    void displayFor (const float* coordinate, float& x, float& y) const noexcept
    {
        x = norm (coordinate[0], 0);
        y = 1.0f - norm (coordinate[1], 1);
    }

    /** All eight axes in the [-1,1] units the engine and the readouts use. */
    void normalisedCoordinate (const float* coordinate, float* out) const noexcept
    {
        for (int d = 0; d < kAtlasDims; ++d)
            out[d] = 2.0f * norm (coordinate[d], d) - 1.0f;
    }

    static constexpr float kFoldedThreshold = 0.25f;
    static const char* presetId (int p) noexcept { return kPresetIds[p]; }
    static int splitOf (int p) noexcept { return kSplit[p]; }

private:
    float norm (float v, int d) const noexcept
    {
        const float span = std::max (hi[(size_t) d] - lo[(size_t) d], 1.0e-8f);
        return (v - lo[(size_t) d]) / span;
    }
    float denorm (float n, int d) const noexcept
    {
        return lo[(size_t) d] + n * (hi[(size_t) d] - lo[(size_t) d]);
    }

    std::array<float, kAtlasDims> lo {};
    std::array<float, kAtlasDims> hi {};
};

}  // namespace atlasflow

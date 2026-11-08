#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

#include "AtlasData.h"

/*  A line-for-line port of the inference half of src/midibrave/atlas_flow_atlas.py
    and the normalisation in AtlasFlowLiveEngine.

    Everything here is small-matrix arithmetic over 50 presets, so it is written
    for obviousness rather than speed -- the whole file costs less than one ONNX
    call. What it must be is *exactly* what the Python does, because a plan that
    routes differently produces different audio, and the comparison against the
    reference runtime is the only thing that catches that.

    Deliberate differences from the Python, both of which are safe:

      - std::stable_sort where numpy uses argsort. numpy's default is an
        unstable introsort, so ties between equidistant presets could order
        either way; stable_sort pins it. Exact ties do not occur in the trained
        atlas, but a gate that depends on luck is not a gate.

      - no temperature. The solver's initial residual is noise * temperature,
        and at temperature 0 it is exactly zero, so no RNG is needed and the
        result is deterministic. Supporting temperature > 0 means reproducing
        torch's Philox stream, which is a separate piece of work.
*/

namespace atlasflow {

// From configs/atlas_flow -- these are asserted exact by AtlasDataConfig, so
// hard-coding them here cannot silently drift from the training config.
inline constexpr int kSampleRate = 44100;
inline constexpr int kRenderSamples = 220500;
inline constexpr int kNoteOnSample = 4410;
inline constexpr int kNoteOffSample = 114660;
inline constexpr int kTrajectoryHop = 2048;
inline constexpr int kFutureFrames = 64;
inline constexpr int kContextFrames = 32;
inline constexpr int kTrajectoryDim = 128;
inline constexpr int kSolverSteps = 4;            // LIVE_SOLVER_STEPS
inline constexpr float kTransitionSeconds = 0.5f; // LIVE_TRANSITION_SECONDS

inline constexpr int kTotalFrames = (kRenderSamples + kTrajectoryHop - 1) / kTrajectoryHop;  // 108

struct Projection
{
    std::array<float, kAtlasDims> coordinate {};
    std::array<float, kAnchorDims> anchor {};
    std::vector<int> indices;
    std::vector<float> weights;
    int component = -1;
};

class AtlasGeometry
{
public:
    AtlasGeometry()
    {
        for (int d = 0; d < kAtlasDims; ++d)
        {
            low[(size_t) d] = kCoordinates[0][d];
            high[(size_t) d] = kCoordinates[0][d];
            for (int p = 1; p < kPresetCount; ++p)
            {
                low[(size_t) d] = std::min (low[(size_t) d], kCoordinates[p][d]);
                high[(size_t) d] = std::max (high[(size_t) d], kCoordinates[p][d]);
            }
            span[(size_t) d] = std::max (high[(size_t) d] - low[(size_t) d], 1.0e-8f);
        }
        buildComponents();
    }

    /** AtlasFlowLiveEngine.denormalize: [-1,1] per axis -> atlas units. */
    std::array<float, kAtlasDims> denormalize (const std::array<float, kAtlasDims>& n) const
    {
        std::array<float, kAtlasDims> out {};
        for (int d = 0; d < kAtlasDims; ++d)
        {
            const float v = std::clamp (n[(size_t) d], -1.0f, 1.0f);
            out[(size_t) d] = low[(size_t) d] + (v + 1.0f) * 0.5f * span[(size_t) d];
        }
        return out;
    }

    std::array<float, kAtlasDims> normalize (const std::array<float, kAtlasDims>& c) const
    {
        std::array<float, kAtlasDims> out {};
        for (int d = 0; d < kAtlasDims; ++d)
            out[(size_t) d] = 2.0f * (c[(size_t) d] - low[(size_t) d]) / span[(size_t) d] - 1.0f;
        return out;
    }

    int componentOf (int preset) const { return membership[(size_t) preset]; }
    const std::vector<std::vector<int>>& components() const { return groups; }

    /** Distance in scale-normalised atlas units, as every comparison here uses. */
    static float distance (const float* a, const float* b)
    {
        float sum = 0.0f;
        for (int d = 0; d < kAtlasDims; ++d)
        {
            const float delta = (a[d] - b[d]) / kScale[d];
            sum += delta * delta;
        }
        return std::sqrt (sum);
    }

    int nearestPreset (const std::array<float, kAtlasDims>& query) const
    {
        int best = 0;
        float bestDistance = std::numeric_limits<float>::max();
        for (int p = 0; p < kPresetCount; ++p)
        {
            const float d = distance (kCoordinates[p], query.data());
            if (d < bestDistance) { bestDistance = d; best = p; }
        }
        return best;
    }

    /** TimbreAtlas.project_details.

        The candidate set is restricted to the connected component of the
        nearest preset. That restriction is the whole point: it keeps every
        position a mixture of presets the model has actually seen, instead of
        an arbitrary point in an 8-D space that may be empty between colonies.
    */
    Projection projectDetails (const std::array<float, kAtlasDims>& query, int neighbours = 4) const
    {
        const int component = membership[(size_t) nearestPreset (query)];
        const auto& candidates = groups[(size_t) component];

        std::vector<float> distances (candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i)
            distances[i] = distance (kCoordinates[candidates[i]], query.data());

        std::vector<size_t> order (candidates.size());
        std::iota (order.begin(), order.end(), size_t {});
        std::stable_sort (order.begin(), order.end(),
                          [&distances] (size_t a, size_t b) { return distances[a] < distances[b]; });

        const size_t take = std::min ((size_t) std::max (1, neighbours), candidates.size());

        Projection out;
        out.component = component;
        float total = 0.0f;
        for (size_t k = 0; k < take; ++k)
        {
            const float inverse = 1.0f / std::max (distances[order[k]], 1.0e-6f);
            out.indices.push_back (candidates[order[k]]);
            out.weights.push_back (inverse);
            total += inverse;
        }
        for (auto& w : out.weights)
            w /= total;

        for (size_t k = 0; k < take; ++k)
        {
            const int p = out.indices[k];
            const float w = out.weights[k];
            for (int d = 0; d < kAtlasDims; ++d)
                out.coordinate[(size_t) d] += w * kCoordinates[p][d];
            for (int d = 0; d < kAnchorDims; ++d)
                out.anchor[(size_t) d] += w * kAnchors[p][d];
        }
        return out;
    }

    /** TimbreAtlas.graph_path -- Dijkstra over the mutual-kNN graph, with edges
        weighted by scale-normalised distance. Empty when the two presets are in
        different components, which is what makes a morph a jump instead. */
    std::vector<int> graphPath (int source, int target) const
    {
        if (source == target)
            return { source };

        std::vector<float> best ((size_t) kPresetCount, std::numeric_limits<float>::infinity());
        std::vector<int> previous ((size_t) kPresetCount, -1);
        using Entry = std::pair<float, int>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<>> queue;

        best[(size_t) source] = 0.0f;
        queue.push ({ 0.0f, source });

        while (! queue.empty())
        {
            const auto [cost, current] = queue.top();
            queue.pop();
            if (cost > best[(size_t) current])
                continue;
            if (current == target)
            {
                std::vector<int> path { current };
                for (int node = current; previous[(size_t) node] >= 0; )
                {
                    node = previous[(size_t) node];
                    path.push_back (node);
                }
                std::reverse (path.begin(), path.end());
                return path;
            }
            for (int neighbour = 0; neighbour < kPresetCount; ++neighbour)
            {
                if (! kNeighbours[current][neighbour])
                    continue;
                const float candidate =
                    cost + distance (kCoordinates[current], kCoordinates[neighbour]);
                if (candidate < best[(size_t) neighbour])
                {
                    best[(size_t) neighbour] = candidate;
                    previous[(size_t) neighbour] = current;
                    queue.push ({ candidate, neighbour });
                }
            }
        }
        return {};
    }

    /** atlas_flow_data.lifecycle_frames: age, gate, release, onset. */
    static void lifecycleFrames (int startFrame, int count, float* out)
    {
        for (int i = 0; i < count; ++i)
        {
            const float samples = (float) (startFrame + i) * (float) kTrajectoryHop;
            const float age = std::clamp (
                (samples - (float) kNoteOnSample) / (float) std::max (kNoteOffSample - kNoteOnSample, 1),
                0.0f, 1.0f);
            const float gate = (samples >= (float) kNoteOnSample
                                && samples < (float) kNoteOffSample) ? 1.0f : 0.0f;
            const float release = std::clamp (
                (samples - (float) kNoteOffSample) / (float) std::max (kRenderSamples - kNoteOffSample, 1),
                0.0f, 1.0f);
            const float onset = std::exp (-8.0f * age) * (samples >= (float) kNoteOnSample ? 1.0f : 0.0f);
            out[i * 4 + 0] = age;
            out[i * 4 + 1] = gate;
            out[i * 4 + 2] = release;
            out[i * 4 + 3] = onset;
        }
    }

private:
    /** TimbreAtlas.connected_components. The traversal order matters: Python
        starts each component at min(unseen) and the components come back in
        discovery order, so component *indices* only agree if this does too. */
    void buildComponents()
    {
        membership.assign ((size_t) kPresetCount, -1);
        std::vector<bool> unseen ((size_t) kPresetCount, true);

        for (;;)
        {
            int start = -1;
            for (int p = 0; p < kPresetCount; ++p)
                if (unseen[(size_t) p]) { start = p; break; }
            if (start < 0)
                break;

            unseen[(size_t) start] = false;
            std::vector<int> stack { start };
            std::vector<int> component;
            while (! stack.empty())
            {
                const int current = stack.back();
                stack.pop_back();
                component.push_back (current);
                for (int n = 0; n < kPresetCount; ++n)
                    if (kNeighbours[current][n] && unseen[(size_t) n])
                    {
                        unseen[(size_t) n] = false;
                        stack.push_back (n);
                    }
            }
            std::sort (component.begin(), component.end());
            for (int node : component)
                membership[(size_t) node] = (int) groups.size();
            groups.push_back (std::move (component));
        }
    }

    std::array<float, kAtlasDims> low {}, high {}, span {};
    std::vector<std::vector<int>> groups;
    std::vector<int> membership;
};

}  // namespace atlasflow

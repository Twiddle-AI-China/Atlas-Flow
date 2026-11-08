#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "AtlasGeometry.h"

/*  The plan: a coordinate on the map becomes five seconds of audio.

    This is the C++ half of AtlasFlowLiveSession._compute_plan_locked. The two
    networks live in ONNX; everything between them is here -- routing across the
    kNN graph, the smoothstep navigation along that route, the lifecycle signal,
    and the Heun solver loop that turns noise into a trajectory.

    The solver is deliberately outside the graph. Four steps of two evaluations
    is pure control flow with no learned parameters, and keeping it here means
    the plan can be interrupted, re-seeded or re-routed without re-exporting
    anything.
*/

namespace atlasflow {

/** The expensive, pitch-independent half of a plan.

    The flow predictor never sees MIDI -- verified bit-exact: the same
    coordinate and seed give an identical trajectory at every note -- so this
    can be computed once per map position and decoded at as many pitches as
    the host asks for. That is what makes a chord affordable and a repeated
    note cheap.
*/
struct Trajectory
{
    std::vector<float> values;                       // [128 * 108], channel-major
    std::array<float, kAtlasDims> coordinate {};
    std::array<float, kAnchorDims> anchor {};
    int component = -1, sourcePreset = -1, targetPreset = -1;
    std::vector<int> route;
    double milliseconds = 0.0;
};

struct Plan
{
    std::vector<float> trajectory;   // [128 * 108], channel-major as the decoder wants
    std::vector<float> audio;        // [220500] mono
    int note = -1;
    double decodeMilliseconds = 0.0;
    std::array<float, kAtlasDims> projectedCoordinate {};
    int component = -1;
    int sourcePreset = -1;
    int targetPreset = -1;
    std::vector<int> route;
    double planMilliseconds = 0.0;
};

class Planner
{
public:
    Planner (const std::string& flowPath, const std::string& decoderPath, int threads = 4)
        : env (ORT_LOGGING_LEVEL_WARNING, "atlasflow")
    {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads (threads);
        options.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
        flow = std::make_unique<Ort::Session> (env, flowPath.c_str(), options);
        decoder = std::make_unique<Ort::Session> (env, decoderPath.c_str(), options);
        flowInputs = names (*flow, true);
        flowOutputs = names (*flow, false);
        decoderInputs = names (*decoder, true);
        decoderOutputs = names (*decoder, false);
    }

    /** Where a morph starts from: what the voice is currently sounding. */
    struct Source
    {
        std::array<float, kAtlasDims> coordinate {};
        std::array<float, kAnchorDims> anchor {};
        int component = -1;
    };

    /** One full plan, from a normalised [-1,1] map position.

        `restart` is a fresh note-on: there is nothing audible to travel from,
        so the plan begins where it is going and the route collapses to a point.
        Passing a `source` instead makes it a morph, and the route machinery
        below walks the kNN graph to get there.

        A move between two different components is not routable -- there is no
        path -- so it is treated as a restart, which is what makes a cross-colony
        move a jump rather than a glide.
    */
    /** Flow only: everything up to but not including the decoder. */
    Trajectory planTrajectory (const std::array<float, kAtlasDims>& normalized,
                               bool restart = true, const Source* source = nullptr,
                               float temperature = 0.0f, uint64_t seed = 20260822)
    {
        const auto started = std::chrono::steady_clock::now();

        const auto query = atlas.denormalize (normalized);
        const auto target = atlas.projectDetails (query);

        const bool crossComponent = source != nullptr
                                 && source->component >= 0
                                 && source->component != target.component;
        const bool fromTarget = restart || crossComponent || source == nullptr;

        const auto sourceCoordinate = fromTarget ? target.coordinate : source->coordinate;
        const auto sourceAnchor = fromTarget ? target.anchor : source->anchor;

        Trajectory out;
        out.component = target.component;
        out.coordinate = target.coordinate;
        out.anchor = target.anchor;
        out.sourcePreset = atlas.nearestPreset (sourceCoordinate);
        out.targetPreset = atlas.nearestPreset (target.coordinate);
        // The Python routes whenever the move stays inside one component, and
        // that includes a restart -- where source == target makes the path a
        // single node anyway, so this stays faithful without a special case.
        out.route = crossComponent ? std::vector<int> {}
                                   : atlas.graphPath (out.sourcePreset, out.targetPreset);

        // Route through the intermediate presets, as the Python does: the
        // endpoints are the projected coordinates, not the presets themselves.
        std::vector<std::array<float, kAtlasDims>> routeCoordinates { sourceCoordinate };
        std::vector<std::array<float, kAnchorDims>> routeAnchors { sourceAnchor };
        for (size_t i = 1; i + 1 < out.route.size(); ++i)
        {
            const int p = out.route[i];
            std::array<float, kAtlasDims> c {};
            std::array<float, kAnchorDims> a {};
            for (int d = 0; d < kAtlasDims; ++d) c[(size_t) d] = kCoordinates[p][d];
            for (int d = 0; d < kAnchorDims; ++d) a[(size_t) d] = kAnchors[p][d];
            routeCoordinates.push_back (c);
            routeAnchors.push_back (a);
        }
        routeCoordinates.push_back (target.coordinate);
        routeAnchors.push_back (target.anchor);

        std::vector<float> edges (routeCoordinates.size() - 1);
        bool anyEdge = false;
        for (size_t i = 0; i + 1 < routeCoordinates.size(); ++i)
        {
            edges[i] = AtlasGeometry::distance (routeCoordinates[i].data(),
                                                routeCoordinates[i + 1].data());
            anyEdge = anyEdge || edges[i] > 1.0e-8f;
        }
        if (! anyEdge)
            std::fill (edges.begin(), edges.end(), 1.0f);

        std::vector<float> cumulative (edges.size() + 1, 0.0f);
        for (size_t i = 0; i < edges.size(); ++i)
            cumulative[i + 1] = cumulative[i] + edges[i];
        const float arc = std::max (cumulative.back(), 1.0e-8f);
        for (auto& v : cumulative)
            v /= arc;

        const int transitionFrames = std::max (2, (int) std::lround (
            kTransitionSeconds * (float) kSampleRate / (float) kTrajectoryHop));

        // Rolling history, seeded from the source anchor with only the last
        // frame unmasked -- the model is told "this is where you are, you have
        // no past", which is what a fresh plan means.
        std::vector<float> history ((size_t) kContextFrames * kTrajectoryDim);
        std::vector<float> historyAnchor (history.size());
        for (int f = 0; f < kContextFrames; ++f)
            for (int d = 0; d < kTrajectoryDim; ++d)
                history[(size_t) f * kTrajectoryDim + d] = sourceAnchor[(size_t) d];
        historyAnchor = history;

        std::vector<char> mask ((size_t) kContextFrames, 0);
        mask[(size_t) kContextFrames - 1] = 1;

        std::vector<float> anchorPath ((size_t) kFutureFrames * kTrajectoryDim);
        std::vector<float> atlasPath ((size_t) kFutureFrames * kAtlasDims);
        std::vector<float> lifecycle ((size_t) kFutureFrames * 4);
        std::vector<float> generated (anchorPath.size());

        std::vector<float> frames;
        frames.reserve ((size_t) kTotalFrames * kTrajectoryDim);

        for (int produced = 0; produced < kTotalFrames; produced += kFutureFrames)
        {
            navigation (produced, transitionFrames, routeCoordinates, routeAnchors,
                        cumulative, edges, anchorPath, atlasPath);
            AtlasGeometry::lifecycleFrames (produced, kFutureFrames, lifecycle.data());

            // The python runtime seeds per chunk as seed + produced; mirrored so a
            // given seed means the same thing on both sides.
            solve (history, historyAnchor, anchorPath, atlasPath, lifecycle, mask,
                   generated, temperature, seed + (uint64_t) produced);

            for (int f = 0; f < kFutureFrames; ++f)
                for (int d = 0; d < kTrajectoryDim; ++d)
                    frames.push_back (generated[(size_t) f * kTrajectoryDim + d]);

            // Slide the window: the newest 32 frames of what we just made
            // become the history for the next chunk, and the mask opens fully.
            rollHistory (history, generated);
            rollHistory (historyAnchor, anchorPath);
            std::fill (mask.begin(), mask.end(), 1);
        }

        // [frames][dim] -> [dim][frames], which is the decoder's layout.
        out.values.assign ((size_t) kTrajectoryDim * kTotalFrames, 0.0f);
        for (int f = 0; f < kTotalFrames; ++f)
            for (int d = 0; d < kTrajectoryDim; ++d)
                out.values[(size_t) d * kTotalFrames + f] =
                    frames[(size_t) f * kTrajectoryDim + d];

        out.milliseconds = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now() - started).count();
        return out;
    }

    /** Decoder only: one pitch from an already-computed trajectory. */
    Plan render (const Trajectory& trajectory, int note)
    {
        const auto started = std::chrono::steady_clock::now();
        Plan out;
        out.trajectory = trajectory.values;
        out.projectedCoordinate = trajectory.coordinate;
        out.component = trajectory.component;
        out.sourcePreset = trajectory.sourcePreset;
        out.targetPreset = trajectory.targetPreset;
        out.route = trajectory.route;
        out.note = note;
        out.audio = decode (trajectory.values, note);
        out.decodeMilliseconds = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now() - started).count();
        out.planMilliseconds = trajectory.milliseconds + out.decodeMilliseconds;
        return out;
    }

    /** Both halves, for callers that do not cache. */
    Plan plan (const std::array<float, kAtlasDims>& normalized, int note,
               bool restart = true, const Source* source = nullptr,
               float temperature = 0.0f, uint64_t seed = 20260822)
    {
        return render (planTrajectory (normalized, restart, source, temperature, seed), note);
    }

    const AtlasGeometry& geometry() const { return atlas; }

private:
    static std::vector<std::string> names (Ort::Session& session, bool inputs)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
        std::vector<std::string> out;
        for (size_t i = 0; i < count; ++i)
        {
            auto held = inputs ? session.GetInputNameAllocated (i, allocator)
                               : session.GetOutputNameAllocated (i, allocator);
            out.emplace_back (held.get());
        }
        return out;
    }

    static std::vector<const char*> raw (const std::vector<std::string>& v)
    {
        std::vector<const char*> out;
        out.reserve (v.size());
        for (const auto& s : v) out.push_back (s.c_str());
        return out;
    }

    /** Smoothstep along the route by arc length, matching the Python exactly:
        the first `transitionFrames` frames travel from source to target and
        everything after sits at the target. Zero endpoint derivatives are the
        reason a morph starts and stops without a discontinuity in rate. */
    static void navigation (int produced, int transitionFrames,
                            const std::vector<std::array<float, kAtlasDims>>& routeCoordinates,
                            const std::vector<std::array<float, kAnchorDims>>& routeAnchors,
                            const std::vector<float>& cumulative,
                            const std::vector<float>& edges,
                            std::vector<float>& anchorPath,
                            std::vector<float>& atlasPath)
    {
        for (int i = 0; i < kFutureFrames; ++i)
        {
            const float index = (float) (produced + i);
            const float phase = std::clamp (index / (float) std::max (transitionFrames - 1, 1), 0.0f, 1.0f);
            const float progress = phase * phase * (3.0f - 2.0f * phase);

            // numpy searchsorted(..., side='right') - 1, clamped into range.
            int edge = 0;
            while (edge + 1 < (int) cumulative.size() && cumulative[(size_t) (edge + 1)] <= progress)
                ++edge;
            edge = std::min ((int) edges.size() - 1, std::max (0, edge));

            const float width = std::max (cumulative[(size_t) edge + 1] - cumulative[(size_t) edge], 1.0e-8f);
            const float alpha = (progress - cumulative[(size_t) edge]) / width;

            for (int d = 0; d < kAtlasDims; ++d)
                atlasPath[(size_t) i * kAtlasDims + d] =
                    routeCoordinates[(size_t) edge][(size_t) d] * (1.0f - alpha)
                    + routeCoordinates[(size_t) edge + 1][(size_t) d] * alpha;
            for (int d = 0; d < kAnchorDims; ++d)
                anchorPath[(size_t) i * kAnchorDims + d] =
                    routeAnchors[(size_t) edge][(size_t) d] * (1.0f - alpha)
                    + routeAnchors[(size_t) edge + 1][(size_t) d] * alpha;
        }
    }

    static void rollHistory (std::vector<float>& history, const std::vector<float>& added)
    {
        // Keep the last kContextFrames of (history ++ added).
        const int keep = kContextFrames;
        std::vector<float> merged;
        merged.reserve (history.size() + added.size());
        merged.insert (merged.end(), history.begin(), history.end());
        merged.insert (merged.end(), added.begin(), added.end());
        const size_t total = merged.size() / kTrajectoryDim;
        const size_t from = total - (size_t) keep;
        std::copy (merged.begin() + (long) (from * kTrajectoryDim), merged.end(), history.begin());
    }

    /** Gaussian noise for the solver's initial residual.

        At temperature 0 this is never called and the result is bit-identical
        to the Python runtime, which is what the comparison tool checks. Above
        zero the two necessarily diverge: matching torch would mean
        reimplementing its Philox stream, and there is no musical reason to --
        a different draw from the same distribution is an equally valid sample
        of the same model. The seed still makes it reproducible here, which is
        what a user actually wants from a seed.
    */
    static float gaussian (uint64_t& state) noexcept
    {
        auto next = [&state]() noexcept
        {
            state += 0x9E3779B97F4A7C15ull;
            uint64_t z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return (z ^ (z >> 31));
        };
        // Box-Muller; the pair's second value is discarded for simplicity, as
        // this runs 8192 times per plan and is nowhere near the cost floor.
        const double u1 = std::max ((double) (next() >> 11) / 9007199254740992.0, 1.0e-12);
        const double u2 = (double) (next() >> 11) / 9007199254740992.0;
        return (float) (std::sqrt (-2.0 * std::log (u1)) * std::cos (6.283185307179586 * u2));
    }

    /** AtlasConditionedTrajectoryFlow.sample. Four Heun steps, two evaluations
        each except the last, which is Euler. */
    void solve (const std::vector<float>& history,
                const std::vector<float>& historyAnchor,
                const std::vector<float>& anchorPath,
                const std::vector<float>& atlasPath,
                const std::vector<float>& lifecycle,
                const std::vector<char>& mask,
                std::vector<float>& outTrajectory,
                float temperature, uint64_t seed)
    {
        std::vector<float> residual (anchorPath.size(), 0.0f);
        if (temperature > 0.0f)
        {
            uint64_t state = seed;
            for (auto& value : residual)
                value = gaussian (state) * temperature;
        }
        std::vector<float> estimate (residual.size());
        std::vector<float> v0 (residual.size()), v1 (residual.size());

        for (int step = 0; step < kSolverSteps; ++step)
        {
            const float t0 = (float) step / (float) kSolverSteps;
            const float t1 = (float) (step + 1) / (float) kSolverSteps;
            const float dt = t1 - t0;

            evaluate (residual, t0, history, historyAnchor, anchorPath, atlasPath, lifecycle, mask, v0);
            for (size_t i = 0; i < residual.size(); ++i)
                estimate[i] = residual[i] + dt * v0[i];

            if (step + 1 < kSolverSteps)
            {
                evaluate (estimate, t1, history, historyAnchor, anchorPath, atlasPath, lifecycle, mask, v1);
                for (size_t i = 0; i < residual.size(); ++i)
                    residual[i] += 0.5f * dt * (v0[i] + v1[i]);
            }
            else
            {
                residual.swap (estimate);
            }
        }

        for (size_t i = 0; i < residual.size(); ++i)
            outTrajectory[i] = anchorPath[i] + residual[i];
    }

    void evaluate (const std::vector<float>& noisyResidual, float flowTime,
                   const std::vector<float>& history,
                   const std::vector<float>& historyAnchor,
                   const std::vector<float>& anchorPath,
                   const std::vector<float>& atlasPath,
                   const std::vector<float>& lifecycle,
                   const std::vector<char>& mask,
                   std::vector<float>& out)
    {
        auto memory = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 3> future { 1, kFutureFrames, kTrajectoryDim };
        const std::array<int64_t, 3> context { 1, kContextFrames, kTrajectoryDim };
        const std::array<int64_t, 3> atlasShape { 1, kFutureFrames, kAtlasDims };
        const std::array<int64_t, 3> lifeShape { 1, kFutureFrames, 4 };
        const std::array<int64_t, 1> timeShape { 1 };
        const std::array<int64_t, 2> maskShape { 1, kContextFrames };

        float time = flowTime;
        auto mutableOf = [] (const std::vector<float>& v) { return const_cast<float*> (v.data()); };

        std::vector<Ort::Value> inputs;
        inputs.push_back (Ort::Value::CreateTensor<float> (memory, mutableOf (noisyResidual), noisyResidual.size(), future.data(), future.size()));
        inputs.push_back (Ort::Value::CreateTensor<float> (memory, &time, 1, timeShape.data(), timeShape.size()));
        inputs.push_back (Ort::Value::CreateTensor<float> (memory, mutableOf (history), history.size(), context.data(), context.size()));
        inputs.push_back (Ort::Value::CreateTensor<float> (memory, mutableOf (historyAnchor), historyAnchor.size(), context.data(), context.size()));
        inputs.push_back (Ort::Value::CreateTensor<float> (memory, mutableOf (anchorPath), anchorPath.size(), future.data(), future.size()));
        inputs.push_back (Ort::Value::CreateTensor<float> (memory, mutableOf (atlasPath), atlasPath.size(), atlasShape.data(), atlasShape.size()));
        inputs.push_back (Ort::Value::CreateTensor<float> (memory, mutableOf (lifecycle), lifecycle.size(), lifeShape.data(), lifeShape.size()));
        inputs.push_back (Ort::Value::CreateTensor<bool> (memory, reinterpret_cast<bool*> (const_cast<char*> (mask.data())), mask.size(), maskShape.data(), maskShape.size()));

        const auto inNames = raw (flowInputs);
        const auto outNames = raw (flowOutputs);
        auto result = flow->Run (Ort::RunOptions { nullptr }, inNames.data(), inputs.data(),
                                 inputs.size(), outNames.data(), outNames.size());
        const float* produced = result.front().GetTensorData<float>();
        std::copy (produced, produced + out.size(), out.begin());
    }

public:
    std::vector<float> decode (const std::vector<float>& trajectory, int note)
    {
        auto memory = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 3> shape { 1, kTrajectoryDim, kTotalFrames };
        const std::array<int64_t, 1> scalar { 1 };
        int64_t noteValue = note;
        int64_t windowStart = 0;

        std::vector<Ort::Value> inputs;
        inputs.push_back (Ort::Value::CreateTensor<float> (memory, const_cast<float*> (trajectory.data()), trajectory.size(), shape.data(), shape.size()));
        inputs.push_back (Ort::Value::CreateTensor<int64_t> (memory, &noteValue, 1, scalar.data(), scalar.size()));
        inputs.push_back (Ort::Value::CreateTensor<int64_t> (memory, &windowStart, 1, scalar.data(), scalar.size()));

        const auto inNames = raw (decoderInputs);
        const auto outNames = raw (decoderOutputs);
        auto result = decoder->Run (Ort::RunOptions { nullptr }, inNames.data(), inputs.data(),
                                    inputs.size(), outNames.data(), outNames.size());
        const auto info = result.front().GetTensorTypeAndShapeInfo();
        const size_t count = info.GetElementCount();
        const float* produced = result.front().GetTensorData<float>();
        return { produced, produced + count };
    }

    Ort::Env env;
    std::unique_ptr<Ort::Session> flow, decoder;
    std::vector<std::string> flowInputs, flowOutputs, decoderInputs, decoderOutputs;
    AtlasGeometry atlas;
};

}  // namespace atlasflow

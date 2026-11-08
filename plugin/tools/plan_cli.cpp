/*  Render one plan from the command line, so the C++ port can be compared
    against the Python runtime sample for sample.

    This is the thing that makes the port trustworthy. The two networks are
    identical by construction (same ONNX graphs), so any difference in the
    output comes from the parts that were hand-ported -- the routing, the
    navigation, the lifecycle signal, the solver loop -- which is exactly what
    needs checking.

        atlasflow-plan --models plugin/models --x 0.2 --y -0.4 --note 50 \
                       --out /tmp/cpp.f32
*/

#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <string>

#include "../src/Planner.h"
#include "../src/Voice.h"

int main (int argc, char** argv)
{
    std::string models = "plugin/models", out;
    float x = 0.0f, y = 0.0f;
    int note = 50, threads = 4, fromPreset = -1;
    int holdBlocks = 0, tailBlocks = 0;   // in LIVE_BLOCK_SAMPLES units
    int polyphony = 0;

    for (int i = 1; i < argc - 1; ++i)
    {
        const std::string flag = argv[i];
        if (flag == "--models") models = argv[++i];
        else if (flag == "--out") out = argv[++i];
        else if (flag == "--x") x = std::stof (argv[++i]);
        else if (flag == "--y") y = std::stof (argv[++i]);
        else if (flag == "--note") note = std::stoi (argv[++i]);
        else if (flag == "--threads") threads = std::stoi (argv[++i]);
        else if (flag == "--from-preset") fromPreset = std::stoi (argv[++i]);
        else if (flag == "--hold-blocks") holdBlocks = std::stoi (argv[++i]);
        else if (flag == "--tail-blocks") tailBlocks = std::stoi (argv[++i]);
        else if (flag == "--poly") polyphony = std::stoi (argv[++i]);
    }

    try
    {
        atlasflow::Planner planner (models + "/flow.onnx", models + "/decoder.onnx", threads);

        std::array<float, atlasflow::kAtlasDims> position {};
        position[0] = x;
        position[1] = y;

        // --from-preset makes this a morph rather than a note-on, which is what
        // exercises the kNN routing: without a source the route is one node.
        atlasflow::Planner::Source source;
        const bool morph = fromPreset >= 0;
        if (morph)
        {
            for (int d = 0; d < atlasflow::kAtlasDims; ++d)
                source.coordinate[(size_t) d] = atlasflow::kCoordinates[fromPreset][d];
            for (int d = 0; d < atlasflow::kAnchorDims; ++d)
                source.anchor[(size_t) d] = atlasflow::kAnchors[fromPreset][d];
            source.component = planner.geometry().componentOf (fromPreset);
        }

        const auto plan = planner.plan (position, note, ! morph, morph ? &source : nullptr);

        double sum = 0.0, peak = 0.0;
        for (float v : plan.audio)
        {
            sum += (double) v * v;
            peak = std::max (peak, (double) std::abs (v));
        }
        const double rms = std::sqrt (sum / (double) plan.audio.size());

        std::printf ("component   %d\n", plan.component);
        std::printf ("preset      %s (index %d)\n",
                     atlasflow::kPresetIds[plan.targetPreset], plan.targetPreset);
        std::printf ("route       %zu node(s):", plan.route.size());
        for (int node : plan.route) std::printf (" %d", node);
        std::printf ("\n");
        std::printf ("trajectory  %zu values\n", plan.trajectory.size());
        std::printf ("audio       %zu samples  rms %.6f  peak %.6f\n",
                     plan.audio.size(), rms, peak);
        std::printf ("plan        %.0f ms total  (flow %.0f + decode %.0f)\n",
                     plan.planMilliseconds,
                     plan.planMilliseconds - plan.decodeMilliseconds,
                     plan.decodeMilliseconds);

        // What a chord costs once the trajectory is in hand: the flow is paid
        // once, the decoder once per voice.
        if (polyphony > 1)
        {
            const auto traj = planner.planTrajectory (position, ! morph, morph ? &source : nullptr);
            const auto t0 = std::chrono::steady_clock::now();
            for (int v = 0; v < polyphony; ++v)
                (void) planner.decode (traj.values, note + v * 4);
            const double total = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now() - t0).count();
            std::printf ("chord       %d voices from one trajectory: %.0f ms "
                         "(%.0f ms/voice, flow %.0f ms paid once)\n",
                         polyphony, total, total / polyphony, traj.milliseconds);
        }

        // --hold-blocks drives the playback layer instead of dumping the raw
        // plan: hold a note through the sustain loop, release it, let the tail
        // run. Blocks are LIVE_BLOCK_SAMPLES so the Python can be matched
        // block for block.
        std::vector<float> played;
        if (holdBlocks > 0)
        {
            constexpr int kBlock = 4096;
            atlasflow::Voice voice;
            voice.accept (&plan, true);
            std::vector<float> block ((size_t) kBlock);
            for (int b = 0; b < holdBlocks; ++b)
            {
                voice.render (block.data(), kBlock);
                played.insert (played.end(), block.begin(), block.end());
            }
            voice.noteOff();
            for (int b = 0; b < tailBlocks; ++b)
            {
                voice.render (block.data(), kBlock);
                played.insert (played.end(), block.begin(), block.end());
            }
            double psum = 0.0;
            for (float v : played) psum += (double) v * v;
            std::printf ("played      %zu samples  rms %.6f\n",
                         played.size(), std::sqrt (psum / (double) played.size()));
        }

        if (! out.empty())
        {
            const auto& dump = played.empty() ? plan.audio : played;
            std::ofstream file (out, std::ios::binary);
            file.write (reinterpret_cast<const char*> (dump.data()),
                        (std::streamsize) (dump.size() * sizeof (float)));
            std::ofstream traj (out + ".traj", std::ios::binary);
            traj.write (reinterpret_cast<const char*> (plan.trajectory.data()),
                        (std::streamsize) (plan.trajectory.size() * sizeof (float)));
            std::printf ("wrote       %s (+ .traj)\n", out.c_str());
        }
    }
    catch (const std::exception& error)
    {
        std::fprintf (stderr, "failed: %s\n", error.what());
        return 1;
    }
    return 0;
}

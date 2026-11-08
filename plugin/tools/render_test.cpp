/*  Headless check that the MIDI path works.

    A plugin that builds proves nothing about whether it makes the right sound,
    and the scaffold's whole purpose is that the MIDI path can be judged before
    the decoder exists. This renders notes offline and asserts on pitch, octave
    folding, the voice ceiling and whether the map actually changes the timbre.

    Built as the AtlasFlowRenderTest target; run it with no arguments.

    `--snapshot <dir>` instead paints the editor offline, once per palette, and
    writes the PNGs there. The surface is the half of this plugin that cannot be
    asserted into correctness -- the checks below can say the terrain has relief
    and the analyser scales a tone properly, and still be looking at a window
    with everything stacked in one corner. This renders it without a window
    server or a screenshot, so it can be looked at on any machine and attached
    to a review.
*/

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>

/*  setenv/unsetenv are POSIX and MSVC has neither; _putenv_s is the Windows
    spelling, and assigning an empty value there is what removes a variable.
    Two lines so the test reads the same on both platforms.
*/
#if defined (_WIN32)
 #define ATLASFLOW_SETENV(name, value) _putenv_s (name, value)
 #define ATLASFLOW_UNSETENV(name)      _putenv_s (name, "")
#else
 #define ATLASFLOW_SETENV(name, value) ::setenv (name, value, 1)
 #define ATLASFLOW_UNSETENV(name)      ::unsetenv (name)
#endif

#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>

#include "../src/Palette.h"
#include "../src/PluginEditor.h"
#include "../src/PluginProcessor.h"
#include "../src/Scopes.h"
#include "../src/Terrain.h"

namespace {

constexpr double kRate = 44100.0;
constexpr int kBlock = 512;

int failures = 0;
bool chordPending = false;

void check (bool ok, const juce::String& what, const juce::String& detail = {})
{
    std::printf ("  %s  %-46s %s\n", ok ? "PASS" : "FAIL",
                 what.toRawUTF8(), detail.toRawUTF8());
    if (! ok) ++failures;
}

/** Render `seconds` of audio with the given notes held throughout. */
std::vector<float> render (atlasflow::AtlasFlowProcessor& p,
                           const std::vector<int>& notes,
                           double seconds,
                           float atlasX = 0.5f, float atlasY = 0.5f)
{
    if (auto* x = p.state().getParameter ("atlasX")) x->setValueNotifyingHost (atlasX);
    if (auto* y = p.state().getParameter ("atlasY")) y->setValueNotifyingHost (atlasY);

    // Offline: the planner needs wall-clock time the render does not spend, so
    // the processor is told this is not real time and may block for a plan.
    p.setNonRealtime (true);
    p.prepareToPlay (kRate, kBlock);
    juce::AudioBuffer<float> buffer (2, kBlock);

    const int totalBlocks = (int) (seconds * kRate / kBlock);
    std::vector<float> out;
    out.reserve ((size_t) (totalBlocks * kBlock));

    for (int b = 0; b < totalBlocks; ++b)
    {
        juce::MidiBuffer midi;
        if (b == 0)
            for (int n : notes)
                midi.addEvent (juce::MidiMessage::noteOn (1, n, (juce::uint8) 100), 0);

        buffer.clear();
        p.processBlock (buffer, midi);
        const auto* ch = buffer.getReadPointer (0);
        out.insert (out.end(), ch, ch + kBlock);
    }
    return out;
}

float rms (const std::vector<float>& x, size_t from = 0)
{
    double sum = 0.0;
    for (size_t i = from; i < x.size(); ++i) sum += (double) x[i] * x[i];
    const size_t n = x.size() - from;
    return n ? (float) std::sqrt (sum / (double) n) : 0.0f;
}

/** Dominant frequency by FFT peak, over the tail (past the attack). */
float dominantHz (const std::vector<float>& x)
{
    constexpr int order = 14;                 // 16384 bins
    constexpr int size = 1 << order;
    if ((int) x.size() < size) return 0.0f;

    juce::dsp::FFT fft (order);
    std::vector<float> buf ((size_t) size * 2, 0.0f);
    const size_t start = x.size() - (size_t) size;
    for (int i = 0; i < size; ++i)
    {
        const float w = 0.5f * (1.0f - std::cos (6.28318530718f * (float) i / (float) (size - 1)));
        buf[(size_t) i] = x[start + (size_t) i] * w;
    }
    fft.performFrequencyOnlyForwardTransform (buf.data());

    int peak = 1;
    for (int i = 2; i < size / 2; ++i)
        if (buf[(size_t) i] > buf[(size_t) peak]) peak = i;
    return (float) peak * (float) kRate / (float) size;
}

float expectedHz (int note) { return 440.0f * std::pow (2.0f, (float) (note - 69) / 12.0f); }

/** Energy at a frequency, in dB relative to the spectrum's own peak.

    The trained model is a pad, and its second harmonic is as strong as its
    fundamental -- measured against the Python runtime, H1 is -0.2 dB and H2 is
    0.0 dB -- so "the FFT peak is the fundamental" is a coin flip and a useless
    assertion. Asking instead whether the fundamental is *present and strong*
    is the question that actually distinguishes a right note from a wrong one.
*/
float harmonicDb (const std::vector<float>& x, float frequency)
{
    constexpr int order = 14;
    constexpr int size = 1 << order;
    if ((int) x.size() < size) return -200.0f;

    juce::dsp::FFT fft (order);
    std::vector<float> buf ((size_t) size * 2, 0.0f);
    const size_t start = x.size() - (size_t) size;
    for (int i = 0; i < size; ++i)
    {
        const float w = 0.5f * (1.0f - std::cos (6.28318530718f * (float) i / (float) (size - 1)));
        buf[(size_t) i] = x[start + (size_t) i] * w;
    }
    fft.performFrequencyOnlyForwardTransform (buf.data());

    float peak = 0.0f;
    for (int i = 1; i < size / 2; ++i) peak = std::max (peak, buf[(size_t) i]);

    const int bin = (int) std::lround (frequency * (float) size / (float) kRate);
    float here = 0.0f;
    for (int i = std::max (1, bin - 3); i <= bin + 3 && i < size / 2; ++i)
        here = std::max (here, buf[(size_t) i]);

    return 20.0f * std::log10 (std::max (here, 1.0e-12f) / std::max (peak, 1.0e-12f));
}

}  // namespace

/** Paint the editor into an image, in each palette, and write them out. */
int snapshot (const juce::File& directory)
{
    juce::ScopedJuceInitialiser_GUI gui;
    if (! directory.createDirectory().wasOk())
    {
        std::printf ("cannot write to %s\n", directory.getFullPathName().toRawUTF8());
        return 1;
    }

    for (int id = 0; id < atlasflow::kPaletteCount; ++id)
    {
        // A fresh processor and editor per palette. Setting the palette on a
        // live editor is what the settings menu does, and it clears the map's
        // cached ground and the scopes' painted strips as it goes; starting
        // clean here means the snapshot cannot accidentally show one palette's
        // history under another's foreground.
        atlasflow::AtlasFlowProcessor processor;
        processor.setAppearance (atlasflow::appearanceKeys::theme, id);
        // Offline, or the scopes draw silence: a plan takes ~500 ms of wall
        // clock and three seconds of blocks run in a fiftieth of that, so in
        // real-time mode nothing would ever land before the picture is taken.
        processor.setNonRealtime (true);
        processor.prepareToPlay (kRate, kBlock);

        // Constructed directly rather than through createEditor(), so the
        // frame hook below is reachable.
        atlasflow::AtlasFlowEditor editor (processor);
        editor.setSize (1200, 760);
        juce::Image shot (juce::Image::ARGB, editor.getWidth(), editor.getHeight(), true);

        // Something has to be sounding, or every scope draws its noise floor
        // and the snapshot says nothing about whether they work.
        juce::AudioBuffer<float> buffer (2, kBlock);
        juce::MidiBuffer midi;
        midi.addEvent (juce::MidiMessage::noteOn (1, 50, 1.0f), 0);
        if (auto* x = processor.state().getParameter ("atlasX")) x->setValueNotifyingHost (0.42f);
        if (auto* y = processor.state().getParameter ("atlasY")) y->setValueNotifyingHost (0.55f);

        const int blocks = (int) (kRate * 5.0 / kBlock);
        for (int block = 0; block < blocks; ++block)
        {
            processor.processBlock (buffer, midi);
            midi.clear();
            // Drive the editor at roughly its own 30 Hz, and *in real time*.
            // Every scope scrolls off the wall clock, so a loop that runs five
            // seconds of audio in a fiftieth of a second advances them by
            // nothing at all and photographs three empty panes.
            if (block % 2 == 0)
            {
                juce::Thread::sleep (16);
                editor.tick();
                // Painted every frame, not only at the end. A scope scrolls
                // *in* paint() -- that is where its strip shifts and its new
                // column lands -- and offline there is no peer, so repaint()
                // schedules nothing. One final paint would photograph three
                // panes that had advanced exactly one column between them.
                juce::Graphics g (shot);
                editor.paintEntireComponent (g, true);
            }
        }

        auto write = [&] (const juce::Image& image, const juce::String& suffix)
        {
            const auto out = directory.getChildFile ("atlas-flow-"
                + juce::String (atlasflow::palette (id).key) + suffix + ".png");
            out.deleteFile();
            std::unique_ptr<juce::FileOutputStream> stream (out.createOutputStream());
            if (stream == nullptr)
            {
                std::printf ("cannot write %s\n", out.getFullPathName().toRawUTF8());
                return false;
            }
            juce::PNGImageFormat png;
            png.writeImageToStream (image, *stream);
            std::printf ("  wrote %s\n", out.getFileName().toRawUTF8());
            return true;
        };

        if (! write (shot, {}))
            return 1;

        // And again with the gear open. Everything that is not the map lives
        // behind it, so a run that only photographs the closed state leaves the
        // whole settings panel -- eleven controls and their attachments -- with
        // no coverage at all.
        editor.showSettings (true);
        for (int frame = 0; frame < 60; ++frame)
        {
            processor.processBlock (buffer, midi);
            juce::Thread::sleep (16);
            editor.tick();
            juce::Graphics g (shot);
            editor.paintEntireComponent (g, true);
        }
        if (! write (shot, "-settings"))
            return 1;
    }
    return 0;
}

int main (int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
        if (juce::String (argv[i]) == "--snapshot" && i + 1 < argc)
        {
            const juce::String where { argv[i + 1] };
            return snapshot (juce::File::isAbsolutePath (where)
                                 ? juce::File (where)
                                 : juce::File::getCurrentWorkingDirectory().getChildFile (where));
        }

    juce::ScopedJuceInitialiser_GUI juceInit;
    std::printf ("Atlas Flow plugin -- headless MIDI render check\n\n");

    // These cases are about the MIDI path -- pitch, folding, voice allocation --
    // which the mock exercises in a fraction of the time a real plan takes.
    // The model engine gets its own section at the end.
    ATLASFLOW_SETENV ("ATLASFLOW_MODELS", "none");
    std::printf ("mock engine\n");

    // 1. A note produces sound at the right pitch.
    {
        atlasflow::AtlasFlowProcessor p;
        const auto audio = render (p, { 60 }, 1.5);
        const float level = rms (audio, audio.size() / 2);
        const float hz = dominantHz (audio);
        const float want = expectedHz (60);
        check (level > 0.01f, "note 60 produces audio",
               juce::String ("rms ") + juce::String (level, 4));
        check (std::abs (hz - want) < want * 0.03f, "note 60 is at the right pitch",
               juce::String (hz, 1) + " Hz, expected " + juce::String (want, 1));
    }

    // 2. Silence when nothing is played.
    {
        atlasflow::AtlasFlowProcessor p;
        const auto audio = render (p, {}, 0.5);
        check (rms (audio) < 1.0e-6f, "no MIDI means silence",
               juce::String ("rms ") + juce::String (rms (audio), 8));
    }

    // 3. Octave folding, the decision from the design doc: a note below the
    //    trained range keeps its pitch class instead of being retuned.
    {
        atlasflow::AtlasFlowProcessor p;
        const auto audio = render (p, { 24 }, 1.5);     // C1 -> folds to C2 (36)
        const float hz = dominantHz (audio);
        const float want = expectedHz (36);
        check (std::abs (hz - want) < want * 0.03f, "note 24 folds up to 36, not clamped",
               juce::String (hz, 1) + " Hz, expected " + juce::String (want, 1));
    }
    {
        atlasflow::AtlasFlowProcessor p;
        const auto audio = render (p, { 84 }, 1.5);     // C6 -> folds down to C4 (60)
        const float hz = dominantHz (audio);
        const float want = expectedHz (60);
        check (std::abs (hz - want) < want * 0.03f, "note 84 folds down to 60",
               juce::String (hz, 1) + " Hz, expected " + juce::String (want, 1));
    }

    // 4. The voice ceiling holds when the host sends more notes than we allow.
    {
        atlasflow::AtlasFlowProcessor p;
        const auto audio = render (p, { 48, 52, 55, 59, 62, 65 }, 1.0);
        const int live = p.activeVoices.load();
        check (live == atlasflow::AtlasFlowProcessor::kMaxVoices,
               "six notes are capped at four voices",
               juce::String (live) + " sounding");
        check (rms (audio, audio.size() / 2) > 0.01f, "a stolen-voice chord still sounds");
    }

    // 5. Moving across the map changes the timbre. Same note, two corners:
    //    the fundamental should stay put while the harmonic content differs.
    {
        atlasflow::AtlasFlowProcessor a, b;
        const auto left = render (a, { 60 }, 1.5, 0.05f, 0.5f);
        const auto right = render (b, { 60 }, 1.5, 0.95f, 0.5f);
        const float hzL = dominantHz (left), hzR = dominantHz (right);
        check (std::abs (hzL - hzR) < 2.0f, "map position does not change pitch",
               juce::String (hzL, 1) + " vs " + juce::String (hzR, 1) + " Hz");

        double diff = 0.0, energy = 0.0;
        const size_t n = juce::jmin (left.size(), right.size());
        for (size_t i = n / 2; i < n; ++i)
        {
            diff += std::abs ((double) left[i] - right[i]);
            energy += std::abs ((double) left[i]);
        }
        const double rel = energy > 0.0 ? diff / energy : 0.0;
        check (rel > 0.05, "map position does change timbre",
               juce::String ("relative waveform difference ") + juce::String (rel * 100.0, 1) + "%");
    }

    // ---- model engine, only when the exported graphs are present ----
    ATLASFLOW_UNSETENV ("ATLASFLOW_MODELS");
    {
        // Each processor holds two ONNX sessions -- about 92 MB of weights plus
        // the runtime's arenas -- so they are created and destroyed one at a
        // time. Holding three at once got this test SIGKILLed on an 8 GB
        // machine with a DAW already open.
        bool haveModel = false;
        {
            atlasflow::AtlasFlowProcessor probe;
            haveModel = probe.modelLoaded();
            if (haveModel)
                check (probe.getLatencySamples() == 0, "latency is zero before prepareToPlay");
        }

        if (! haveModel)
        {
            std::printf ("\nmodel engine: skipped (no graphs; run scripts/atlas_flow/export_onnx.py)\n");
        }
        else
        {
            std::printf ("\nmodel engine\n");
            atlasflow::AtlasFlowProcessor p;
            // A plan takes about 700 ms, and the reported latency is what buys
            // that time, so this renders well past it before judging.
            // Longer than the reported latency, which now scales with voices.
            const auto audio = render (p, { 60 }, 6.0);
            const float level = rms (audio, audio.size() * 3 / 4);
            const float hz = dominantHz (audio);
            const float want = expectedHz (60);
            // Zero, and that is the point. A note-on is served from the
            // pre-decoded bank -- PCM that already exists -- so there is
            // nothing to wait for and nothing for the host to compensate.
            check (p.getLatencySamples() == 0, "reports no latency",
                   juce::String (p.getLatencySamples()) + " samples");

            // The bank fills on a worker in wall-clock time, so an offline test
            // has to wait for it the way a player would.
            //
            // The rate is reported rather than merely waited on, because it is
            // the number that decides how roaming feels: moving the map
            // retires the whole bank, so the refill rate is how long the
            // instrument spends unable to answer a note-on from PCM.
            // Both figures are measured against what the bank held when this
            // started, not against zero. The checks above already asked for
            // plans, so some pitches can be banked before the first sample is
            // taken -- an earlier version of this divided the whole elapsed
            // time by the whole bank and reported a cold-start latency of
            // 0.0002 ms, which is the time it takes to observe work that had
            // already happened.
            const int startedWith = p.bankReady();
            const auto bankStart = std::chrono::steady_clock::now();
            const auto deadline = bankStart + std::chrono::seconds (120);
            double firstMs = -1.0;
            while (p.bankReady() < startedWith + 4 && std::chrono::steady_clock::now() < deadline)
            {
                if (firstMs < 0.0 && p.bankReady() > startedWith)
                    firstMs = std::chrono::duration<double, std::milli> (
                        std::chrono::steady_clock::now() - bankStart).count();
                std::this_thread::sleep_for (std::chrono::milliseconds (5));
            }
            const double bankMs = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now() - bankStart).count();
            const int ready = p.bankReady();
            const int added = ready - startedWith;
            check (ready > 0, "the bank fills itself while idle",
                   juce::String (ready) + " pitches ready, " + juce::String (added) + " added"
                   + (startedWith == 0 && firstMs >= 0.0
                          ? juce::String (", first from cold after ") + juce::String (firstMs, 0) + " ms"
                          : juce::String())
                   + (added > 0 ? ", " + juce::String (bankMs / added, 0) + " ms/pitch"
                                : juce::String (", no rate: none arrived")));

            {
                juce::AudioBuffer<float> probe (2, kBlock);
                juce::MidiBuffer midi;
                midi.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0);
                probe.clear();
                p.processBlock (probe, midi);
                float peak = 0.0f;
                for (int i = 0; i < kBlock; ++i)
                    peak = std::max (peak, std::abs (probe.getReadPointer (0)[i]));
                check (peak > 1.0e-5f, "a banked note sounds in the very first block",
                       juce::String ("peak ") + juce::String (peak, 6)
                       + ", served from bank " + juce::String (p.notesFromBank()));
            }
            check (level > 1.0e-4f, "model note produces audio",
                   juce::String ("rms ") + juce::String (level, 5));
            const float fundamental = harmonicDb (audio, want);
            const float ratio = hz / want;
            const float nearestHarmonic = std::round (ratio);
            check (fundamental > -12.0f, "model note has a strong fundamental",
                   juce::String (want, 1) + " Hz at " + juce::String (fundamental, 1) + " dB");
            check (nearestHarmonic >= 1.0f && std::abs (ratio - nearestHarmonic) < 0.04f,
                   "spectral peak is a harmonic of the note",
                   juce::String (hz, 1) + " Hz = " + juce::String ((int) nearestHarmonic)
                   + "x " + juce::String (want, 1));

            // The whole point of reporting latency: a note-on's audio must come
            // out exactly `latency` samples later, so the host's compensation
            // puts it back on the beat. If the plan time were added on top --
            // which it was, before the ring was addressed by absolute position
            // -- every note in a sequenced part would land late.
            const int latency = p.getLatencySamples();
            int onset = -1;
            for (size_t i = 0; i < audio.size(); ++i)
                if (std::abs (audio[i]) > 1.0e-4f) { onset = (int) i; break; }
            const int drift = onset < 0 ? -1 : std::abs (onset - latency);
            check (onset >= 0 && onset < kBlock * 4,
                   "a note starts immediately, not after a plan",
                   juce::String ("onset ") + juce::String (onset) + ", latency "
                   + juce::String (latency) + ", drift " + juce::String (drift));

            chordPending = true;

        }

        // The symptom reported from real use: a note played while the map is
        // moving sometimes never started. A walking path moves the position
        // every block, and a morph request issued every block overwrote the
        // note-on before the planner ever saw it.
        if (chordPending)
        {
            atlasflow::AtlasFlowProcessor p;
            p.setNonRealtime (true);
            p.prepareToPlay (kRate, kBlock);
            juce::AudioBuffer<float> buffer (2, kBlock);
            auto* px = p.state().getParameter ("atlasX");
            auto* py = p.state().getParameter ("atlasY");

            const int blocks = (int) (8.0 * kRate / kBlock);
            std::vector<float> out;
            for (int b = 0; b < blocks; ++b)
            {
                // Sweep the map continuously, as a drawn path does.
                const float t = (float) b / (float) blocks;
                if (px) px->setValueNotifyingHost (0.2f + 0.6f * t);
                if (py) py->setValueNotifyingHost (0.5f + 0.3f * std::sin (t * 9.0f));

                juce::MidiBuffer midi;
                if (b == 0)
                    midi.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0);
                buffer.clear();
                p.processBlock (buffer, midi);
                const auto* ch = buffer.getReadPointer (0);
                out.insert (out.end(), ch, ch + kBlock);
            }

            int onset = -1;
            for (size_t i = 0; i < out.size(); ++i)
                if (std::abs (out[i]) > 1.0e-4f) { onset = (int) i; break; }
            const int latency = p.getLatencySamples();
            check (onset >= 0, "a note started while the map was sweeping",
                   onset < 0 ? "never sounded" : juce::String ("onset ") + juce::String (onset));
            check (onset >= 0 && onset <= latency + (int) kRate,
                   "and started close to the reported latency",
                   juce::String ("onset ") + juce::String (onset)
                   + ", latency " + juce::String (latency));
            check (p.planCacheHits() > 0, "the sweep reused cached plans",
                   juce::String (p.planCacheHits()) + " hits / "
                   + juce::String (p.planCacheMisses()) + " misses");
        }

        // A chord is one flow and N decodes: every note has to be audible,
        // which is the property the trajectory cache exists to make affordable.
        // Run after the processor above is gone, for the memory reason noted.
        if (chordPending)
        {
            atlasflow::AtlasFlowProcessor chordProcessor;
            const std::vector<int> chord { 48, 52, 55 };
            const auto chordAudio = render (chordProcessor, chord, 9.0);
            bool allPresent = true;
            juce::String detail;
            for (int n : chord)
            {
                const float level = harmonicDb (chordAudio, expectedHz (n));
                allPresent = allPresent && level > -24.0f;
                detail += juce::String (n) + ":" + juce::String (level, 0) + "dB ";
            }
            check (allPresent, "a three-note chord sounds every note", detail);
        }
    }

    // -----------------------------------------------------------------------
    // the frontend
    //
    // The map draws claims -- which presets the model never saw, where the
    // runtime will overrule you, what you are hearing -- and a claim drawn
    // wrongly is worse than one not drawn at all, because it still looks
    // authoritative. These check the claims, not the pixels.
    // -----------------------------------------------------------------------
    std::printf ("\nfrontend\n");
    {
        using namespace atlasflow;

        // --- the held-out split ------------------------------------------
        int validation = 0, test = 0;
        juce::String heldOut;
        for (int p = 0; p < kPresetCount; ++p)
        {
            if (Atlas::splitOf (p) == kValidation) ++validation;
            if (Atlas::splitOf (p) == kTest) { ++test; heldOut += juce::String (Atlas::presetId (p)) + " "; }
        }
        // atlas_flow_manifest.split_presets at SEED 20260822. Hard-coded here
        // on purpose: the rings on the map are only honest if they are around
        // these three, and a regenerated AtlasData.h that moved them would
        // otherwise pass every other check in this file.
        const bool right = validation == 2 && test == 3
                        && heldOut.contains ("serum_s048066")
                        && heldOut.contains ("serum_s043574")
                        && heldOut.contains ("serum_s048543");
        check (right, "the map rings the presets the model never saw",
               juce::String (validation) + " validation, " + juce::String (test) + " test: " + heldOut);

        // --- the terrain --------------------------------------------------
        const auto before = std::chrono::steady_clock::now();
        const auto& terrain = Terrain::shared();
        const double buildMs = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now() - before).count();

        float lowest = 1.0f, highest = 0.0f;
        for (int row = 0; row < Terrain::kGrid; ++row)
            for (int column = 0; column < Terrain::kGrid; ++column)
            {
                lowest = juce::jmin (lowest, terrain.at (row, column));
                highest = juce::jmax (highest, terrain.at (row, column));
            }
        check (highest - lowest > 0.5f, "the terrain has relief to shade",
               juce::String (lowest, 3) + " to " + juce::String (highest, 3)
                   + ", built in " + juce::String (buildMs, 1) + " ms");

        auto heightAt = [&terrain] (float x, float y)
        {
            const int column = juce::roundToInt ((x + 1.0f) * 0.5f * (Terrain::kGrid - 1));
            const int row = juce::roundToInt ((1.0f - y) * 0.5f * (Terrain::kGrid - 1));
            return terrain.at (row, column);
        };

        // A preset alone in its component can only ever resolve to itself, so
        // standing anywhere near it means being overruled. That is the single
        // most important thing the shading says, and it is the one that was
        // wrong in the first version of this field.
        const AtlasGeometry geometry;
        float loneGround = 0.0f, crowdedGround = 0.0f;
        int lone = -1, crowded = -1;
        for (int p = 0; p < kPresetCount; ++p)
        {
            const auto size = geometry.components()[(size_t) geometry.componentOf (p)].size();
            std::array<float, kAtlasDims> raw {};
            for (int d = 0; d < kAtlasDims; ++d) raw[(size_t) d] = kCoordinates[p][d];
            const auto at = geometry.normalize (raw);
            const float ground = heightAt (at[0], at[1]);
            if (size == 1 && (lone < 0 || ground < loneGround)) { lone = p; loneGround = ground; }
            if (size > 8 && (crowded < 0 || ground > crowdedGround)) { crowded = p; crowdedGround = ground; }
        }
        check (lone >= 0 && crowded >= 0 && loneGround < crowdedGround,
               "a singleton preset sits lower than a colony",
               juce::String (Atlas::presetId (lone)) + " " + juce::String (loneGround, 3)
                   + " vs " + juce::String (Atlas::presetId (crowded)) + " " + juce::String (crowdedGround, 3));

        // The desktop build measures the same field with a plain Euclidean
        // metric over its own reimplementation of the projection; this one
        // calls the planner's AtlasGeometry instead. The two have to agree, or
        // the plugin is drawing a different map from the one in the paper.
        {
            std::vector<float> mine, theirs;
            for (int row = 0; row < Terrain::kGrid; row += 4)
                for (int column = 0; column < Terrain::kGrid; column += 4)
                {
                    const float x = (float) column / (Terrain::kGrid - 1) * 2.0f - 1.0f;
                    const float y = 1.0f - (float) row / (Terrain::kGrid - 1) * 2.0f;
                    mine.push_back (Terrain::overrideAt (x, y));

                    // app.js overrideAt(): nearest preset's component, four
                    // nearest within it, inverse-distance, no per-axis scale.
                    const auto lifted = Terrain::lift (x, y);
                    std::array<std::array<float, kAtlasDims>, kPresetCount> normalised {};
                    std::vector<std::pair<float, int>> ranked;
                    for (int p = 0; p < kPresetCount; ++p)
                    {
                        std::array<float, kAtlasDims> raw {};
                        for (int d = 0; d < kAtlasDims; ++d) raw[(size_t) d] = kCoordinates[p][d];
                        normalised[(size_t) p] = geometry.normalize (raw);
                        float sum = 0.0f;
                        for (int d = 0; d < kAtlasDims; ++d)
                        {
                            const float delta = normalised[(size_t) p][(size_t) d] - lifted[(size_t) d];
                            sum += delta * delta;
                        }
                        ranked.push_back ({ sum, p });
                    }
                    std::sort (ranked.begin(), ranked.end());
                    const int home = geometry.componentOf (ranked.front().second);
                    std::array<float, kAtlasDims> blended {};
                    float total = 0.0f;
                    int taken = 0;
                    for (const auto& [squared, p] : ranked)
                    {
                        if (geometry.componentOf (p) != home || taken >= Terrain::kNear)
                            continue;
                        ++taken;
                        const float weight = 1.0f / juce::jmax (std::sqrt (squared), 1.0e-6f);
                        total += weight;
                        for (int d = 0; d < kAtlasDims; ++d)
                            blended[(size_t) d] += normalised[(size_t) p][(size_t) d] * weight;
                    }
                    float moved = 0.0f;
                    for (int d = 0; d < kAtlasDims; ++d)
                    {
                        const float delta = blended[(size_t) d] / total - lifted[(size_t) d];
                        moved += delta * delta;
                    }
                    theirs.push_back (std::sqrt (moved));
                }

            const auto mean = [] (const std::vector<float>& v)
            { return std::accumulate (v.begin(), v.end(), 0.0f) / (float) v.size(); };
            const float mineMean = mean (mine), theirsMean = mean (theirs);
            float cov = 0.0f, varMine = 0.0f, varTheirs = 0.0f;
            for (size_t i = 0; i < mine.size(); ++i)
            {
                cov += (mine[i] - mineMean) * (theirs[i] - theirsMean);
                varMine += (mine[i] - mineMean) * (mine[i] - mineMean);
                varTheirs += (theirs[i] - theirsMean) * (theirs[i] - theirsMean);
            }
            const float correlation = cov / std::sqrt (varMine * varTheirs);
            check (correlation > 0.95f, "the field agrees with the desktop instrument's",
                   "correlation " + juce::String (correlation, 4) + " over "
                       + juce::String ((int) mine.size()) + " points");
        }

        // --- the palettes --------------------------------------------------
        {
            bool allClimb = true;
            juce::String detail;
            for (int id = 0; id < kPaletteCount; ++id)
            {
                const auto& theme = palette (id);
                // A colormap has to be monotone in brightness or the eye reads
                // the same level as two different values. Paper runs the other
                // way -- on a light ground the loud parts get darker -- so
                // either direction counts, but wandering does not.
                const float first = theme.ramp.front().getPerceivedBrightness();
                const float last = theme.ramp.back().getPerceivedBrightness();
                const bool rising = last > first;
                bool monotone = true;
                for (size_t i = 1; i < theme.ramp.size(); ++i)
                {
                    const float step = theme.ramp[i].getPerceivedBrightness()
                                     - theme.ramp[i - 1].getPerceivedBrightness();
                    monotone = monotone && (rising ? step > 0.01f : step < -0.01f);
                }
                const bool readable = std::abs (last - first) > 0.45f
                    && std::abs (theme.ink.getPerceivedBrightness()
                                 - theme.background.getPerceivedBrightness()) > 0.5f;
                allClimb = allClimb && monotone && readable;
                if (! (monotone && readable))
                    detail += juce::String (theme.key) + " ";
            }
            check (allClimb, "every palette's ramp is monotone and legible",
                   detail.isEmpty() ? juce::String (kPaletteCount) + " palettes" : "bad: " + detail);
        }

        // --- the analyser --------------------------------------------------
        //
        // The scopes are tuned against a Web Audio AnalyserNode, so what
        // matters is not that this FFT is correct in general but that it
        // returns what that node returns: Blackman window, magnitudes over
        // fftSize, mapped across -100..-30 dB.
        {
            ScopeTap tap;
            Analyser analyser;
            constexpr float amplitude = 0.01f;
            constexpr float hz = 440.0f;
            std::vector<float> tone (Analyser::kFftSize);
            for (int i = 0; i < Analyser::kFftSize; ++i)
                tone[(size_t) i] = amplitude * std::sin (juce::MathConstants<float>::twoPi
                                                         * hz * (float) i / (float) kRate);
            tap.write (tone.data(), (int) tone.size());
            for (int i = 0; i < 60; ++i)         // past the 0.45 smoothing
                analyser.update (tap);

            const auto& level = analyser.frequency();
            int peak = 0;
            for (int bin = 1; bin < Analyser::kBins; ++bin)
                if (level[(size_t) bin] > level[(size_t) peak]) peak = bin;
            const float peakHz = (float) peak * (float) kRate / (float) Analyser::kFftSize;
            check (std::abs (peakHz - hz) <= (float) kRate / (float) Analyser::kFftSize,
                   "the analyser finds a tone where it is",
                   juce::String (peakHz, 1) + " Hz, expected " + juce::String (hz, 1));

            // A Blackman-windowed sine of amplitude A peaks at A * 0.21 once
            // divided by fftSize, which for 0.01 is -53.6 dB and lands 0.663
            // of the way up the node's own dB window.
            const float expected = (juce::Decibels::gainToDecibels (amplitude * 0.21f) + 100.0f) / 70.0f;
            check (std::abs (level[(size_t) peak] - expected) < 0.04f,
                   "and scales it as an AnalyserNode would",
                   juce::String (level[(size_t) peak], 3) + ", expected " + juce::String (expected, 3));
        }

        // --- the scope views ----------------------------------------------
        {
            ScopeTap tap;
            Analyser analyser;
            std::vector<float> noise (Analyser::kFftSize);
            juce::Random random (7);
            for (auto& sample : noise)
                sample = 0.05f * (random.nextFloat() * 2.0f - 1.0f);
            tap.write (noise.data(), (int) noise.size());
            for (int i = 0; i < 20; ++i)
                analyser.update (tap);

            juce::String empty;
            for (int kind = 0; kind < kScopeKindCount; ++kind)
            {
                ScopeView view;
                view.setKind ((ScopeKind) kind);
                juce::Image pane (juce::Image::ARGB, 220, 90, true);
                const auto& theme = palette (kMono);
                // Twice, so the spectrogram actually scrolls its strip: the
                // in-place memmove is the one piece of this that a single
                // frame would never exercise.
                for (int frame = 0; frame < 2; ++frame)
                {
                    juce::Graphics g (pane);
                    view.paint (g, { 0, 0, 220, 90 }, analyser, theme, 0.2f, kRate, false);
                }
                int drawn = 0;
                for (int y = 0; y < pane.getHeight(); y += 2)
                    for (int x = 0; x < pane.getWidth(); x += 2)
                        if (pane.getPixelAt (x, y) != theme.background)
                            ++drawn;
                if (drawn < 40)
                    empty += juce::String (scopeName ((ScopeKind) kind)) + " ";
            }
            check (empty.isEmpty(), "every scope view draws the signal",
                   empty.isEmpty() ? juce::String (kScopeKindCount) + " views"
                                   : "blank: " + empty);
        }
    }

    std::printf ("\n%s\n", failures == 0 ? "all checks passed"
                                         : juce::String (failures).toRawUTF8());
    return failures == 0 ? 0 : 1;
}

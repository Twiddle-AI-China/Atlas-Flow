#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>

#include "Analyser.h"
#include "Atlas.h"
#include "DrawnPath.h"
#include "MockVoice.h"
#include "PlanService.h"
#include "Voice.h"

namespace atlasflow {

/** Atlas Flow, as a plugin.

    The host owns the MIDI; the map owns the timbre. X and Y are ordinary
    automatable parameters rather than a bespoke gesture, because in a DAW an
    automation lane *is* the path drawn across the atlas in the paper, and it is
    the form a producer already knows how to draw, copy and quantise.

    Two engines live here. When the exported graphs are present the real model
    runs, planned on a worker thread and played back from pre-rendered PCM. When
    they are not, the three-waveform mock takes over so the plugin still loads
    and still makes a navigable sound -- which is what keeps the UI and the MIDI
    path workable while the model is being changed.

    Latency is the price of the real engine. A plan costs around 700 ms and
    cannot run in the audio callback, so the output is delayed by a fixed amount
    and that amount is reported to the host. The host's delay compensation then
    puts a *sequenced* part back on time; live playing still feels the wait, and
    the proposal says so rather than pretending otherwise.
*/
class AtlasFlowProcessor final : public juce::AudioProcessor
{
public:
    static constexpr int kMaxVoices = 4;

    /** Reported latency for a given voice count, from this machine's own speed.

        A chord costs one flow and N decodes -- the flow predictor never sees
        MIDI, so a trajectory is cached per map position and shared across the
        chord. The latency has to cover that, because a plan that misses its
        deadline is a note the host has already asked for and will not get.

        The timings are measured at load rather than hard-coded, because the
        same binary runs about 1.8x slower under Rosetta than natively (655 ms
        against 1187 ms for one plan here) and a constant sized from the native
        figure is a deadline the translated build cannot meet. A host with
        Rosetta enabled is not a hypothetical: it is what this plugin was first
        installed into.

        Block-streaming the decoder would collapse all of it -- 41 ms per 16384
        samples puts four voices inside real time, and the exported graph
        already takes the window_start input that needs.
    */
    double latencyFor (int voices) const;

    AtlasFlowProcessor();
    ~AtlasFlowProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout&) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Atlas Flow"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 3.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return "Default"; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    juce::AudioProcessorValueTreeState& state() noexcept { return parameters; }
    const Atlas& atlas() const noexcept { return map; }

    /** What the editor's scopes read. Written from the audio thread, drained by
        the editor at frame rate; see ScopeTap for why it is a tap and not a
        queue. Public because it is the editor's own business, and routing it
        through an accessor would say otherwise. */
    ScopeTap scope;

    /** Appearance, which is not automation.

        The palette and the three pane assignments belong to the session, not
        to a mix: automating them would record a lane full of a producer
        changing their mind about colours. They live as properties on the APVTS
        tree instead, which getStateInformation already serialises whole, so
        they are saved with the project and travel with a preset without
        appearing in any automation list.
    */
    int appearance (const juce::Identifier& key, int fallback) const;
    void setAppearance (const juce::Identifier& key, int value);

    bool modelLoaded() const noexcept { return service != nullptr; }
    unsigned planCacheHits() const { return service ? service->hits() : 0; }
    unsigned notesFromBank() const { return bankServed.load (std::memory_order_relaxed); }
    unsigned notesMissedBank() const { return bankMissed.load (std::memory_order_relaxed); }
    int bankReady() const { return service ? service->bankReady() : 0; }
    uint64_t bankIdentityForPosition() const;
    unsigned planCacheMisses() const { return service ? service->misses() : 0; }
    juce::String engineDescription() const;

    /** The drawn path. Set from the editor on mouse-up; the audio thread takes
        its own copy under the lock once and then reads lock-free. Clearing it
        hands control back to the X/Y parameters. */
    void setPath (std::vector<juce::Point<float>> stroke);
    void clearPath();
    /** For the editor to draw: a snapshot of the stroke and where the walk is. */
    std::vector<juce::Point<float>> pathPoints() const;
    float pathPhase() const noexcept { return walkPhase.load (std::memory_order_relaxed); }
    bool pathActive() const noexcept { return hasPath.load (std::memory_order_relaxed); }

    // Read by the editor. Written from the audio thread, so relaxed atomics.
    std::atomic<int>   audiblePreset { 0 };
    std::atomic<int>   activeVoices  { 0 };
    std::atomic<float> audibleSine   { 0.0f };
    std::atomic<float> audibleSaw    { 0.0f };
    std::atomic<float> audibleSquare { 0.0f };
    std::atomic<bool>  planning      { false };
    // Where the sounding voice actually is, in display coordinates, after the
    // runtime has projected the request onto legal territory. The gap between
    // this and the X/Y parameters is the override made visible -- the same
    // quantity the terrain shades, for the one point you are standing on.
    std::atomic<float> voiceX { -1.0f }, voiceY { -1.0f };
    /** The same point on all eight axes, for the inset. PC1 and PC2 hold only
        42.3% and 19.8% of the variance between them, so the other six are most
        of what distinguishes one sound from another and the map cannot draw
        any of it. The inset is where they get said out loud. */
    std::array<std::atomic<float>, kAtlasDims> voiceCoordinate {};
    std::atomic<unsigned> bankServed { 0 }, bankMissed { 0 };

private:
    void handleNoteOn (int note, float velocity, int noteOnOffset = 0);
    void handleNoteOff (int note);
    void renderSegment (float* out, int start, int count);
    void requestPlan (bool restart);
    std::array<float, kAtlasDims> currentPosition() const;

    static juce::AudioProcessorValueTreeState::ParameterLayout layout();
    static juce::String findModels();

    juce::AudioProcessorValueTreeState parameters;
    std::atomic<float>* paramX = nullptr;
    std::atomic<float>* paramY = nullptr;
    std::atomic<float>* paramLevel = nullptr;

    Atlas map;

    // --- drawn path -------------------------------------------------------
    PathData incomingPath;            // written by the editor, under pathLock
    PathData audioPath;               // the audio thread's own copy
    PathWalker walker;
    juce::CriticalSection pathLock;
    std::atomic<bool> pathPending { false };
    std::atomic<bool> hasPath { false };
    std::atomic<float> walkPhase { 0.0f };
    juce::Point<float> walkPosition { 0.5f, 0.5f };

    // --- real engine -----------------------------------------------------
    struct ModelVoice
    {
        Voice voice;
        int note = -1;
        float velocity = 1.0f;
        long long writePosition = 0;   // absolute position of its next sample
        long long noteOnPosition = 0;  // where the note being planned begins
        uint32_t stamp = 0;
        bool sounding = false;
    };

    std::unique_ptr<PlanService> service;
    std::array<ModelVoice, kMaxVoices> modelVoices;
    int configuredVoices = kMaxVoices;
    float lastRequestedX = -2.0f, lastRequestedY = -2.0f;
    int settleBlocks = 0;
    static constexpr int kSettleBlocks = 12;   // ~0.14 s at 512/44100
    std::vector<float> modelScratch;

    // Output ring addressed by absolute sample position, so a plan that lands
    // late can still be written where it belongs instead of where "now" is.
    // Voices *add* into it, so a chord sums correctly even when its notes'
    // plans finish at different times.
    std::vector<float> ring;
    long long absolutePosition = 0;   // samples since prepareToPlay
    long long ringZeroedTo = 0;       // ring is cleared and accumulating up to here
    int delaySamples = 0;

    // --- mock engine -----------------------------------------------------
    std::array<MockVoice, kMaxVoices> voices;
    uint32_t noteCounter = 0;
    juce::SmoothedValue<float> mixSine, mixSaw, mixSquare;
    std::vector<float> rampSine, rampSaw, rampSquare;
    std::array<float, kPresetCount> weights {};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AtlasFlowProcessor)
};

}  // namespace atlasflow

#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace atlasflow {

juce::AudioProcessorValueTreeState::ParameterLayout AtlasFlowProcessor::layout()
{
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout l;

    /*  How each parameter writes itself down.

        Without this a float parameter prints at full precision -- "0.8000000",
        "10.00000" -- and that is not only the plugin's own panel: it is what
        the host's generic editor, its automation lane and its parameter list
        all show. A unit and a sensible number of decimals belong to the
        parameter, so they are stated once here rather than in each place that
        happens to display one.
    */
    auto places = [] (int decimals, String unit)
    {
        return AudioParameterFloatAttributes().withStringFromValueFunction (
            [decimals, unit] (float value, int) { return String (value, decimals) + unit; });
    };
    l.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "atlasX", 1 }, "Atlas X", NormalisableRange<float> { 0.0f, 1.0f }, 0.5f,
        places (2, {})));
    l.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "atlasY", 1 }, "Atlas Y", NormalisableRange<float> { 0.0f, 1.0f }, 0.5f,
        places (2, {})));
    l.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "level", 1 }, "Level", NormalisableRange<float> { 0.0f, 1.0f }, 0.8f,
        places (2, {})));
    // Voices against latency, stated plainly in the names because that is what
    // the choice actually costs.
    l.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "morph", 1 }, "Morph",
        NormalisableRange<float> { 0.5f, 5.0f }, 2.0f, places (1, " s")));
    // Above zero this stops being bit-identical to the python runtime by
    // design: matching torch would mean reimplementing its Philox stream, and
    // a different draw from the same distribution is an equally valid sample.
    l.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "temperature", 1 }, "Temperature",
        NormalisableRange<float> { 0.0f, 1.0f }, 0.0f, places (2, {})));
    l.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "seed", 1 }, "Seed",
        NormalisableRange<float> { 0.0f, 9999.0f, 1.0f }, 822.0f, places (0, {})));
    l.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "lap", 1 }, "Lap",
        NormalisableRange<float> { 2.0f, 60.0f }, 10.0f, places (1, " s")));
    l.add (std::make_unique<AudioParameterChoice> (
        ParameterID { "voices", 1 }, "Voices",
        StringArray { "1 (0.6s latency)", "2 (1.05s)", "3 (1.5s)", "4 (1.95s)" }, 3));
    return l;
}

/** Where the exported graphs live.

    Checked in order so a developer can point at a build tree without installing
    anything, while a shipped plugin finds its own copy. Absent all of them the
    processor falls back to the mock engine rather than refusing to load -- a
    plugin that vanishes from the host's list is far harder to diagnose than one
    that loads and says it has no model.
*/
juce::String AtlasFlowProcessor::findModels()
{
    juce::StringArray candidates;
    const auto* fromEnvironment = std::getenv ("ATLASFLOW_MODELS");
    // An empty value counts as unset. Windows removes a variable by assigning
    // "" to it, and some shells leave the empty string behind instead -- which
    // the pointer check alone would read as "look in this directory called
    // nothing", silently disabling the model engine.
    if (fromEnvironment != nullptr && *fromEnvironment != '\0')
    {
        // An explicit setting wins outright and does not fall through to the
        // built-in locations: "look here" that silently looks elsewhere is the
        // kind of thing that wastes an afternoon. "none" forces the mock.
        const juce::String value (fromEnvironment);
        if (value == "none")
            return {};
        candidates.add (value);
    }
    else
    {
        // Inside our own bundle first: that is the copy that travels with the
        // plugin, and the only one that exists on someone else's machine.
        const auto binary = juce::File::getSpecialLocation (juce::File::currentExecutableFile);
        // Contents/MacOS on macOS, Contents/x86_64-win on Windows -- a VST3 is
        // a bundle on both, so the same two steps up land on Contents.
        candidates.add (binary.getParentDirectory()
                              .getParentDirectory()
                              .getChildFile ("Resources").getChildFile ("models")
                              .getFullPathName());
        // A Windows standalone .exe has no bundle around it, so the graphs sit
        // beside it. On macOS this is Contents/MacOS, where they never are, so
        // it costs one stat and finds nothing.
        candidates.add (binary.getParentDirectory().getChildFile ("models").getFullPathName());
#ifdef ATLASFLOW_DEFAULT_MODELS
        candidates.add (ATLASFLOW_DEFAULT_MODELS);
#endif
        candidates.add (juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                            .getChildFile ("AtlasFlow").getChildFile ("models").getFullPathName());
    }

    for (const auto& candidate : candidates)
    {
        const juce::File directory (candidate);
        if (directory.getChildFile ("flow.onnx").existsAsFile()
            && directory.getChildFile ("decoder.onnx").existsAsFile())
            return directory.getFullPathName();
    }
    return {};
}

AtlasFlowProcessor::AtlasFlowProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      parameters (*this, nullptr, "ATLASFLOW", layout())
{
    paramX = parameters.getRawParameterValue ("atlasX");
    paramY = parameters.getRawParameterValue ("atlasY");
    paramLevel = parameters.getRawParameterValue ("level");

    const auto models = findModels();
    if (models.isNotEmpty())
    {
        try
        {
            service = std::make_unique<PlanService> (
                (models + "/flow.onnx").toStdString(),
                (models + "/decoder.onnx").toStdString());
        }
        catch (const std::exception&)
        {
            // A corrupt or mismatched graph should degrade to the mock, not
            // take the host down with it.
            service.reset();
        }
    }
}

AtlasFlowProcessor::~AtlasFlowProcessor() = default;

uint64_t AtlasFlowProcessor::bankIdentityForPosition() const
{
    // Must agree with PlanService's own hash, or a filled bank never matches.
    return service == nullptr ? 0
         : PlanService::identityFor (currentPosition(),
                                     parameters.getRawParameterValue ("temperature")->load(),
                                     (uint64_t) parameters.getRawParameterValue ("seed")->load());
}

void AtlasFlowProcessor::setPath (std::vector<juce::Point<float>> stroke)
{
    PathData built;
    built.build (std::move (stroke));
    const bool ok = built.valid();
    {
        const juce::ScopedLock lock (pathLock);
        incomingPath = std::move (built);
    }
    pathPending.store (true, std::memory_order_release);
    hasPath.store (ok, std::memory_order_relaxed);
}

void AtlasFlowProcessor::clearPath()
{
    {
        const juce::ScopedLock lock (pathLock);
        incomingPath = {};
    }
    pathPending.store (true, std::memory_order_release);
    hasPath.store (false, std::memory_order_relaxed);
}

std::vector<juce::Point<float>> AtlasFlowProcessor::pathPoints() const
{
    const juce::ScopedLock lock (pathLock);
    return incomingPath.points;
}

juce::String AtlasFlowProcessor::engineDescription() const
{
    return service != nullptr ? "MODEL" : "MOCK";
}

double AtlasFlowProcessor::latencyFor (int voices) const
{
    const int n = juce::jmax (1, voices);
    // Fall back to figures measured natively on an M2 if the probe has not run;
    // a translated host will measure itself slower and get a longer, correct
    // latency rather than a deadline it cannot keep.
    double flow = 0.11, decode = 0.40;
    if (service != nullptr && service->measuredDecodeSeconds() > 0.0)
    {
        flow = service->measuredFlowSeconds();
        decode = service->measuredDecodeSeconds();
    }
    // A quarter of margin: the measurement is one sample of a figure that moves
    // with whatever else the machine is doing.
    return (flow + decode * (double) n) * 1.25;
}

void AtlasFlowProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    for (auto& v : voices)
        v.prepare (sampleRate);

    const auto capacity = (size_t) juce::jmax (1, samplesPerBlock);
    rampSine.assign (capacity, 0.0f);
    rampSaw.assign (capacity, 0.0f);
    rampSquare.assign (capacity, 0.0f);
    modelScratch.assign (capacity, 0.0f);

    const double ramp = 0.03;
    mixSine.reset (sampleRate, ramp);
    mixSaw.reset (sampleRate, ramp);
    mixSquare.reset (sampleRate, ramp);

    if (service != nullptr)
    {
        // The delay is what makes the reported latency real: a note-on gets
        // this much wall clock before its audio has to leave the buffer, which
        // is the whole mechanism that puts a sequenced part back on time.
        // Block until the load-time probe has measured this machine, and let
        // it get a head start on the bank.
        service->waitUntilWarm();
        configuredVoices = juce::jlimit (1, kMaxVoices,
            1 + (int) parameters.getParameterAsValue ("voices").getValue());

        // Zero. A note-on is served from the pre-decoded bank, which is PCM
        // that already exists, so there is nothing to wait for and nothing to
        // compensate. Reporting a latency the instrument does not need would
        // push every other track in the session around for no reason.
        //
        // When a pitch is not in the bank yet the note starts as soon as it is
        // rendered -- late, but only that note. That is a far better failure
        // than delaying the whole instrument by the worst case.
        delaySamples = 0;
        ring.assign (2 * capacity + 1, 0.0f);
        absolutePosition = 0;
        ringZeroedTo = 0;
        for (auto& v : modelVoices) v = {};
        service->prime (currentPosition(),
                        parameters.getRawParameterValue ("temperature")->load(),
                        (uint64_t) parameters.getRawParameterValue ("seed")->load());
        setLatencySamples (delaySamples);
    }
    else
    {
        delaySamples = 0;
        ring.clear();
        setLatencySamples (0);
    }

    lastRequestedX = lastRequestedY = -2.0f;
}

bool AtlasFlowProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
}

std::array<float, kAtlasDims> AtlasFlowProcessor::currentPosition() const
{
    // The map's two axes run 0..1 as parameters and -1..1 in the atlas; the
    // remaining six dimensions come from the presets the projection lands on,
    // which is what keeps every position a mixture of real presets.
    // A drawn path takes over from the parameters while it is walking, the
    // same way dragging takes over from the pointer in the desktop build. The
    // parameters stay live underneath, so clearing the path hands control
    // straight back to host automation.
    const float x = hasPath.load (std::memory_order_relaxed)
                  ? walkPosition.x : juce::jlimit (0.0f, 1.0f, paramX->load());
    const float y = hasPath.load (std::memory_order_relaxed)
                  ? walkPosition.y : juce::jlimit (0.0f, 1.0f, paramY->load());

    std::array<float, kAtlasDims> out {};
    out[0] = x * 2.0f - 1.0f;
    out[1] = (1.0f - y) * 2.0f - 1.0f;
    return out;
}

void AtlasFlowProcessor::requestPlan (bool restart)
{
    if (service == nullptr)
        return;
    const auto position = currentPosition();
    for (int v = 0; v < configuredVoices; ++v)
    {
        const auto& mv = modelVoices[(size_t) v];
        if (mv.sounding)
            service->request (v, position, mv.note, restart,
                              parameters.getRawParameterValue ("temperature")->load(),
                              (uint64_t) parameters.getRawParameterValue ("seed")->load());
    }
}

void AtlasFlowProcessor::handleNoteOn (int note, float velocity, int noteOnOffset)
{
    if (service != nullptr)
    {
        // Each note is an independent plan: the trajectory is shared across the
        // chord (one flow) but every pitch needs its own decode.
        int chosen = -1;
        for (int v = 0; v < configuredVoices; ++v)
            if (! modelVoices[(size_t) v].sounding) { chosen = v; break; }
        if (chosen < 0)
        {
            uint32_t oldest = 0xffffffffu;
            for (int v = 0; v < configuredVoices; ++v)
                if (modelVoices[(size_t) v].stamp < oldest)
                { oldest = modelVoices[(size_t) v].stamp; chosen = v; }
        }
        if (chosen < 0)
            return;

        auto& mv = modelVoices[(size_t) chosen];
        mv.note = MockVoice::foldToRange (note);
        mv.velocity = velocity;
        mv.noteOnPosition = absolutePosition + noteOnOffset;
        mv.stamp = ++noteCounter;
        mv.sounding = true;
        service->setBankCentre (mv.note);

        // The bank first, and this is the whole point: if the pitch has been
        // rendered the note starts in this very block.
        if (service->bankIdentity() == bankIdentityForPosition())
        {
            if (const Plan* ready = service->fromBank (mv.note))
            {
                mv.voice.accept (ready, true, parameters.getRawParameterValue ("morph")->load());
                mv.writePosition = mv.noteOnPosition;
                bankServed.fetch_add (1, std::memory_order_relaxed);
                return;
            }
        }

        bankMissed.fetch_add (1, std::memory_order_relaxed);
        service->request (chosen, currentPosition(), mv.note, true,
                          parameters.getRawParameterValue ("temperature")->load(),
                          (uint64_t) parameters.getRawParameterValue ("seed")->load());
        return;
    }

    MockVoice* target = nullptr;
    for (auto& v : voices)
        if (! v.active()) { target = &v; break; }
    if (target == nullptr)
    {
        uint32_t oldest = 0xffffffffu;
        for (auto& v : voices)
            if (v.stamp() < oldest) { oldest = v.stamp(); target = &v; }
    }
    if (target != nullptr)
        target->start (note, velocity, ++noteCounter);
}

void AtlasFlowProcessor::handleNoteOff (int note)
{
    if (service != nullptr)
    {
        const int folded = MockVoice::foldToRange (note);
        for (int v = 0; v < configuredVoices; ++v)
        {
            auto& mv = modelVoices[(size_t) v];
            if (mv.sounding && mv.note == folded)
            {
                mv.voice.noteOff();
                mv.sounding = false;
                // Anything still being planned for this voice is now stale; let
                // it arrive and be ignored rather than start a note the player
                // has already released.
                service->cancel (v);
            }
        }
        return;
    }

    const int folded = MockVoice::foldToRange (note);
    for (auto& v : voices)
        if (v.active() && v.currentNote() == folded)
            v.release();
}

void AtlasFlowProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    buffer.clear();

    float pc1 = 0.0f, pc2 = 0.0f;
    map.pointerToAtlas (paramX->load(), paramY->load(), pc1, pc2);
    map.blend (pc1, pc2, weights.data());

    PresetTimbre target { 0.0f, 0.0f, 0.0f };
    for (int p = 0; p < kPresetCount; ++p)
    {
        const float w = weights[(size_t) p];
        if (w <= 0.0f) continue;
        const auto t = timbreFor (p);
        target.sine += w * t.sine;
        target.saw += w * t.saw;
        target.square += w * t.square;
    }
    mixSine.setTargetValue (target.sine);
    mixSaw.setTargetValue (target.saw);
    mixSquare.setTargetValue (target.square);

    audiblePreset.store (map.nearest (pc1, pc2), std::memory_order_relaxed);
    audibleSine.store (target.sine, std::memory_order_relaxed);
    audibleSaw.store (target.saw, std::memory_order_relaxed);
    audibleSquare.store (target.square, std::memory_order_relaxed);

    auto* left = buffer.getWritePointer (0);
    const int total = buffer.getNumSamples();

    int cursor = 0;
    for (const auto meta : midi)
    {
        const int at = juce::jlimit (0, total, meta.samplePosition);
        if (at > cursor)
        {
            renderSegment (left, cursor, at - cursor);
            cursor = at;
        }
        const auto m = meta.getMessage();
        if (m.isNoteOn())       handleNoteOn (m.getNoteNumber(), m.getFloatVelocity(), at);
        else if (m.isNoteOff()) handleNoteOff (m.getNoteNumber());
        else if (m.isAllNotesOff() || m.isAllSoundOff())
        {
            if (service != nullptr)
            {
                for (int v = 0; v < kMaxVoices; ++v)
                {
                    modelVoices[(size_t) v].voice.noteOff();
                    modelVoices[(size_t) v].sounding = false;
                    service->cancel (v);
                }
            }
            else
                for (auto& v : voices) v.release();
        }
    }
    if (cursor < total)
        renderSegment (left, cursor, total - cursor);

    buffer.applyGain (paramLevel->load());

    if (service != nullptr)
    {
        // Pick up a path the editor just handed over. Done once per block and
        // only when something changed, so the lock is effectively never
        // contended; the audio thread then reads its own copy.
        if (pathPending.load (std::memory_order_acquire))
        {
            const juce::ScopedTryLock lock (pathLock);
            if (lock.isLocked())
            {
                audioPath = incomingPath;
                walker.reset();
                pathPending.store (false, std::memory_order_relaxed);
            }
        }

        // Walk the path. This is the gesture the paper is about: the stroke,
        // not the pointer, is what moves the timbre.
        if (hasPath.load (std::memory_order_relaxed) && audioPath.valid())
        {
            walkPosition = walker.advance (audioPath, total, getSampleRate(),
                                           parameters.getRawParameterValue ("lap")->load());
            walkPhase.store (walker.position(), std::memory_order_relaxed);
        }

        // A map move while a note is sounding is a morph, not a retrigger.
        const float x = hasPath.load (std::memory_order_relaxed) ? walkPosition.x : paramX->load();
        const float y = hasPath.load (std::memory_order_relaxed) ? walkPosition.y : paramY->load();
        // One morph in flight per voice, not one per audio block.
        //
        // A walking path moves the position every block, so the old test --
        // "did x or y change at all" -- issued roughly 340 requests a second at
        // a planner that can serve one or two. It never caught up, and the
        // note-on requests queued behind it never got served either. Asking
        // only when the previous request for that voice has been answered rate
        // limits this to whatever the machine can actually do, which is also
        // what makes the sound lag the path by a visible amount rather than
        // stall outright.
        const float moved = std::max (std::abs (x - lastRequestedX), std::abs (y - lastRequestedY));
        if (moved > 0.004f)
        {
            bool issued = false;
            const auto position = currentPosition();
            const float temperature = parameters.getRawParameterValue ("temperature")->load();
            const auto seed = (uint64_t) parameters.getRawParameterValue ("seed")->load();
            for (int v = 0; v < configuredVoices; ++v)
            {
                const auto& mv = modelVoices[(size_t) v];
                if (mv.sounding && ! service->busyFor (v))
                {
                    service->request (v, position, mv.note, false, temperature, seed);
                    issued = true;
                }
            }
            if (issued)
            {
                lastRequestedX = x;
                lastRequestedY = y;
            }
            settleBlocks = 0;
        }
        else if (++settleBlocks == kSettleBlocks)
        {
            // The map has stopped moving: start rendering this sound now rather
            // than at the first key press.
            service->prime (currentPosition(),
                            parameters.getRawParameterValue ("temperature")->load(),
                            (uint64_t) parameters.getRawParameterValue ("seed")->load());
        }
        planning.store (service->busy(), std::memory_order_relaxed);
        int live = 0;
        for (int v = 0; v < configuredVoices; ++v)
            if (modelVoices[(size_t) v].voice.currentState() != Voice::State::idle) ++live;
        activeVoices.store (live, std::memory_order_relaxed);

        // The voice count changes the reported latency, so tell the host when
        // it moves rather than silently reporting a number that no longer holds.
        const int wanted = juce::jlimit (1, kMaxVoices,
            1 + (int) parameters.getParameterAsValue ("voices").getValue());
        if (wanted != configuredVoices)
        {
            configuredVoices = wanted;
            delaySamples = (int) std::lround (latencyFor (wanted) * getSampleRate());
            if ((size_t) delaySamples + 2 * (size_t) total > ring.size())
                ring.assign ((size_t) delaySamples + 2 * (size_t) total, 0.0f);
            setLatencySamples (delaySamples);
            updateHostDisplay (ChangeDetails {}.withLatencyChanged (true));
        }
    }
    else
    {
        int live = 0;
        for (auto& v : voices) if (v.active()) ++live;
        activeVoices.store (live, std::memory_order_relaxed);
    }

    absolutePosition += total;

    // Tap what actually leaves the plugin, after level and after the model's
    // own latency compensation. Tapping before either would draw a picture of
    // something the user is not hearing, which is worse than no picture.
    scope.write (buffer.getReadPointer (0), total);

    for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
        buffer.copyFrom (ch, 0, buffer, 0, 0, total);
}

void AtlasFlowProcessor::renderSegment (float* out, int start, int count)
{
    if (count <= 0)
        return;

    if (service != nullptr)
    {
        // Offline, wait for the planner. A bounce runs faster than real time,
        // so background plans would never land and the render would be silent;
        // there is no deadline to miss offline, so blocking is correct here and
        // would be forbidden in the real-time case.
        if (isNonRealtime())
            service->waitUntilSettled();

        const long long target = absolutePosition + start + count;
        const size_t size = ring.size();

        // Clear forward once, then let every voice add into it. Accumulating
        // rather than overwriting is what lets a chord sum correctly when its
        // notes' plans finish at different moments.
        for (long long pos = ringZeroedTo; pos < target; ++pos)
            ring[(size_t) (pos % (long long) size)] = 0.0f;
        ringZeroedTo = std::max (ringZeroedTo, target);

        for (int v = 0; v < configuredVoices; ++v)
        {
            auto& mv = modelVoices[(size_t) v];

            if (const Plan* fresh = service->tryAcquire (v))
            {
                const bool restart = service->lastWasRestart (v);
                float vx = 0.0f, vy = 0.0f;
                map.displayFor (fresh->projectedCoordinate.data(), vx, vy);
                voiceX.store (vx, std::memory_order_relaxed);
                voiceY.store (vy, std::memory_order_relaxed);
                std::array<float, kAtlasDims> normalised {};
                map.normalisedCoordinate (fresh->projectedCoordinate.data(), normalised.data());
                for (int d = 0; d < kAtlasDims; ++d)
                    voiceCoordinate[(size_t) d].store (normalised[(size_t) d], std::memory_order_relaxed);
                mv.voice.accept (fresh, restart,
                                 parameters.getRawParameterValue ("morph")->load());
                if (restart)
                {
                    // The point of the reported latency: a note-on's audio
                    // belongs at the note-on, not at the moment its plan
                    // happened to finish. That position has not been read yet,
                    // so write back to it rather than appending at "now" --
                    // otherwise the plan time stacks on top of the latency the
                    // host already compensated and every note lands late.
                    const long long earliest = absolutePosition - delaySamples;
                    mv.writePosition = std::max (mv.noteOnPosition, earliest);
                }
            }

            while (mv.writePosition < target)
            {
                const int chunk = (int) std::min<long long> (target - mv.writePosition,
                                                             (long long) modelScratch.size());
                mv.voice.render (modelScratch.data(), chunk, mv.velocity);
                for (int i = 0; i < chunk; ++i)
                    ring[(size_t) ((mv.writePosition + i) % (long long) size)] +=
                        modelScratch[(size_t) i];
                mv.writePosition += chunk;
            }

            const Plan* heldByVoice[3] {};
            mv.voice.heldPlans (heldByVoice);
            service->publishHeld (v, heldByVoice[0], heldByVoice[1], heldByVoice[2]);
        }

        for (int i = 0; i < count; ++i)
        {
            const long long read = absolutePosition + start + i - delaySamples;
            out[start + i] = read < 0 ? 0.0f : ring[(size_t) (read % (long long) size)];
        }
        return;
    }

    if (rampSine.size() < (size_t) count)
    {
        rampSine.resize ((size_t) count);
        rampSaw.resize ((size_t) count);
        rampSquare.resize ((size_t) count);
    }
    for (int i = 0; i < count; ++i)
    {
        rampSine[(size_t) i] = mixSine.getNextValue();
        rampSaw[(size_t) i] = mixSaw.getNextValue();
        rampSquare[(size_t) i] = mixSquare.getNextValue();
    }
    for (auto& v : voices)
        v.render (out + start, count, rampSine.data(), rampSaw.data(), rampSquare.data());
}

int AtlasFlowProcessor::appearance (const juce::Identifier& key, int fallback) const
{
    const auto value = parameters.state.getProperty (key);
    return value.isVoid() ? fallback : (int) value;
}

void AtlasFlowProcessor::setAppearance (const juce::Identifier& key, int value)
{
    parameters.state.setProperty (key, value, nullptr);
}

void AtlasFlowProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = parameters.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void AtlasFlowProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        if (xml->hasTagName (parameters.state.getType()))
            parameters.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessorEditor* AtlasFlowProcessor::createEditor()
{
    return new AtlasFlowEditor (*this);
}

}  // namespace atlasflow

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new atlasflow::AtlasFlowProcessor();
}

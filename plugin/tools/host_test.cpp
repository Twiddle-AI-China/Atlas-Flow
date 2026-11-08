/*  Load the built plugin the way a DAW does, and check what a DAW would see.

    Everything else in this directory tests the processor by constructing it
    directly. That skips the part a user actually installs: the VST3 and AU
    wrappers, the bundle layout, the parameter metadata the host reads, and the
    state blob the host saves into a project. A plugin can pass every one of
    those other checks and still fail to scan.

    So this links no Atlas Flow code at all. It is a host: it scans a bundle,
    instantiates it through juce::AudioPluginFormat, and drives it through the
    generic AudioPluginInstance interface -- the same interface Logic or Live
    has. Anything it can reach is something a host can reach.

        atlasflow-host-test                        # whatever is installed
        atlasflow-host-test --vst3 path/to.vst3    # a specific build output

    AU is always taken from the system registry, because that is the only way
    macOS will load one; the build installs it there, and auval needs it there
    too. On Windows only the VST3 arm runs, which is also the only format
    shipped there.
*/

#include <cstdio>
#include <memory>

#include <juce_audio_utils/juce_audio_utils.h>

namespace {

constexpr double kRate = 44100.0;
constexpr int kBlock = 512;

int failures = 0;

void check (bool ok, const juce::String& what, const juce::String& detail = {})
{
    std::printf ("  %s  %-46s %s\n", ok ? "PASS" : "FAIL",
                 what.toRawUTF8(), detail.toRawUTF8());
    if (! ok) ++failures;
}

float rms (const juce::AudioBuffer<float>& buffer)
{
    double sum = 0.0;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        const float value = buffer.getSample (0, i);
        sum += (double) value * value;
    }
    return (float) std::sqrt (sum / juce::jmax (1, buffer.getNumSamples()));
}

/** How many digits a displayed value carries after the point.

    A float parameter with no string conversion of its own prints at full
    precision, and that text is what the host puts in its generic editor, its
    parameter list and its automation lane. "0.8000000" is not a bug the
    plugin's own window would ever show once it formats its sliders itself,
    which is exactly why it needs checking from out here.
*/
int decimalsIn (const juce::String& text)
{
    const int dot = text.indexOfChar ('.');
    if (dot < 0)
        return 0;
    int digits = 0;
    for (int i = dot + 1; i < text.length() && juce::CharacterFunctions::isDigit (text[i]); ++i)
        ++digits;
    return digits;
}

/** Everything a host can ask of the thing it just loaded. */
void exercise (juce::AudioPluginInstance& plugin, const juce::String& format)
{
    std::printf ("\n%s\n", format.toRawUTF8());

    check (plugin.getName().contains ("Atlas"), "the host reads the right name",
           plugin.getName());
    check (plugin.acceptsMidi(), "it announces itself as taking MIDI");

    plugin.setPlayConfigDetails (0, 2, kRate, kBlock);
    plugin.prepareToPlay (kRate, kBlock);

    check (plugin.getLatencySamples() == 0, "it reports no latency to the host",
           juce::String (plugin.getLatencySamples()) + " samples");

    juce::AudioBuffer<float> buffer (juce::jmax (2, plugin.getTotalNumOutputChannels()), kBlock);
    juce::MidiBuffer midi;

    // Silence first: a wrapper that leaks the previous host's buffer, or that
    // never clears, shows up here and nowhere else.
    buffer.clear();
    plugin.processBlock (buffer, midi);
    check (rms (buffer) < 1.0e-6f, "no MIDI means silence",
           juce::String (rms (buffer), 8));

    // Then a note. The model engine plans on a worker, so this waits the way a
    // player would rather than assuming the first block is already sound.
    midi.addEvent (juce::MidiMessage::noteOn (1, 60, 1.0f), 0);
    float loudest = 0.0f;
    for (int block = 0; block < (int) (kRate * 6.0 / kBlock); ++block)
    {
        buffer.clear();
        plugin.processBlock (buffer, midi);
        midi.clear();
        loudest = juce::jmax (loudest, rms (buffer));
        if (loudest > 1.0e-4f)
            break;
        juce::Thread::sleep (2);
    }
    check (loudest > 1.0e-4f, "a note played through the wrapper sounds",
           "rms " + juce::String (loudest, 6));

    // --- what the host shows for every parameter -------------------------
    const auto& parameters = plugin.getParameters();
    check (! parameters.isEmpty(), "the host can see the parameters",
           juce::String (parameters.size()) + " of them");
    // Seven are real. Anything much past that means a wrapper is publishing
    // something nobody asked for -- which is exactly how 2080 MIDI CC
    // parameters got into the VST3 without anyone noticing.
    check (parameters.size() <= 32, "and is not flooded with them",
           juce::String (parameters.size()) + " parameters");

    juce::String sloppy;
    for (auto* parameter : parameters)
    {
        const auto text = parameter->getText (parameter->getValue(), 32);
        if (decimalsIn (text) > 2)
            sloppy += parameter->getName (32) + " (" + text + ") ";
    }
    check (sloppy.isEmpty(), "every parameter states a readable value",
           sloppy.isEmpty() ? juce::String ("none over 2 decimals") : "runaway: " + sloppy);

    // --- the state blob a project file keeps -----------------------------
    //
    // Round-tripped through the wrapper, not through the processor: the VST3
    // and AU wrappers each have their own path for this, and a preset that
    // comes back wrong is the kind of thing only a host notices.
    if (auto* level = parameters[0])
    {
        const float original = level->getValue();
        level->setValueNotifyingHost (0.33f);

        juce::MemoryBlock saved;
        plugin.getStateInformation (saved);
        check (saved.getSize() > 0, "it hands the host a state blob",
               juce::String ((int) saved.getSize()) + " bytes");

        level->setValueNotifyingHost (0.77f);
        plugin.setStateInformation (saved.getData(), (int) saved.getSize());
        check (std::abs (level->getValue() - 0.33f) < 0.01f,
               "and restores it to what was saved",
               juce::String (level->getValue(), 3) + ", expected 0.330");

        level->setValueNotifyingHost (original);
    }

    // --- the editor, as the wrapper hands it over ------------------------
    check (plugin.hasEditor(), "it offers the host an editor");
    if (auto* editor = plugin.createEditorIfNeeded())
    {
        editor->setSize (900, 600);
        juce::Image shot (juce::Image::ARGB, editor->getWidth(), editor->getHeight(), true);
        {
            juce::Graphics g (shot);
            editor->paintEntireComponent (g, true);
        }
        // Something has to be on it. An editor that constructs and paints
        // nothing passes every other check in this file.
        int drawn = 0;
        for (int y = 0; y < shot.getHeight(); y += 7)
            for (int x = 0; x < shot.getWidth(); x += 7)
                if (shot.getPixelAt (x, y).getAlpha() > 0)
                    ++drawn;
        check (drawn > 1000, "and the editor paints when it is asked",
               juce::String (drawn) + " sampled pixels covered");
        plugin.editorBeingDeleted (editor);
        delete editor;
    }

    plugin.releaseResources();
}

/** Scan, instantiate, and run the checks. Returns false if nothing was found. */
bool run (juce::AudioPluginFormat& format, const juce::String& identifier, const juce::String& label)
{
    juce::OwnedArray<juce::PluginDescription> found;
    format.findAllTypesForFile (found, identifier);
    if (found.isEmpty())
    {
        std::printf ("\n%s\n  FAIL  %-46s %s\n", label.toRawUTF8(),
                     "the host can scan the bundle", identifier.toRawUTF8());
        ++failures;
        return false;
    }

    // Straight to the format rather than through an AudioPluginFormatManager:
    // JUCE 9 resolves this console app against juce_audio_processors_headless,
    // where addDefaultFormats() is deleted, and the manager is only a registry
    // anyway. The format we want is the one already in hand.
    juce::String error;
    std::unique_ptr<juce::AudioPluginInstance> plugin (
        format.createInstanceFromDescription (*found.getFirst(), kRate, kBlock, error));

    if (plugin == nullptr)
    {
        std::printf ("\n%s\n  FAIL  %-46s %s\n", label.toRawUTF8(),
                     "the host can instantiate it", error.toRawUTF8());
        ++failures;
        return false;
    }

    exercise (*plugin, label);
    return true;
}

}  // namespace

int main (int argc, char** argv)
{
    juce::ScopedJuceInitialiser_GUI gui;
    std::printf ("Atlas Flow plugin -- loaded as a host loads it\n");

    juce::String vst3Path;
    for (int i = 1; i < argc; ++i)
        if (juce::String (argv[i]) == "--vst3" && i + 1 < argc)
            vst3Path = argv[i + 1];

    juce::VST3PluginFormat vst3;
    if (vst3Path.isEmpty())
    {
        // Nothing named, so take whatever is installed -- which is what a host
        // would find too.
        const auto installed = vst3.searchPathsForPlugins (vst3.getDefaultLocationsToSearch(), true, false);
        for (const auto& candidate : installed)
            if (candidate.contains ("Atlas Flow"))
                vst3Path = candidate;
    }
    else if (! juce::File::isAbsolutePath (vst3Path))
    {
        vst3Path = juce::File::getCurrentWorkingDirectory().getChildFile (vst3Path).getFullPathName();
    }

    if (vst3Path.isEmpty())
    {
        std::printf ("\nVST3\n  FAIL  %-46s %s\n", "a VST3 to load",
                     "none installed and none given with --vst3");
        ++failures;
    }
    else
    {
        run (vst3, vst3Path, "VST3  " + juce::File (vst3Path).getFileName());
    }

   #if JUCE_PLUGINHOST_AU && JUCE_MAC
    // An AU cannot be loaded from a path: macOS resolves it through the
    // component registry, so the only honest way to find it is to look where a
    // host looks.
    juce::AudioUnitPluginFormat au;
    juce::String auIdentifier;
    for (const auto& candidate : au.searchPathsForPlugins (au.getDefaultLocationsToSearch(), true, false))
        if (candidate.contains ("Atfl") && candidate.contains ("Twdl"))
            auIdentifier = candidate;

    if (auIdentifier.isEmpty())
    {
        std::printf ("\nAU\n  FAIL  %-46s %s\n", "an AU to load",
                     "not registered -- build the AU target first");
        ++failures;
    }
    else
    {
        run (au, auIdentifier, "AU");
    }
   #endif

    std::printf ("\n%s\n", failures == 0 ? "the host is happy"
                                         : (juce::String (failures) + " failed").toRawUTF8());
    return failures == 0 ? 0 : 1;
}

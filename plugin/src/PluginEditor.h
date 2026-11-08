#pragma once

#include <array>
#include <memory>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "Analyser.h"
#include "Palette.h"
#include "PluginProcessor.h"
#include "Scopes.h"
#include "Strings.h"
#include "Terrain.h"

namespace atlasflow {

/** Where the appearance choices are kept on the APVTS tree. */
namespace appearanceKeys {
inline const juce::Identifier theme { "theme" };
inline const juce::Identifier slot0 { "slot0" };
inline const juce::Identifier slot1 { "slot1" };
inline const juce::Identifier slot2 { "slot2" };
inline const juce::Identifier drawerShut { "drawerShut" };
inline const juce::Identifier language { "language" };
}

/** The atlas, drawn.

    One full-bleed surface: the terrain is how much the runtime will overrule
    you, the hairlines are the mutual-kNN graph, every preset is a dot, and the
    pointer is the instrument. Dragging writes the two automation parameters,
    so a gesture here and an automation lane in the host are the same thing.

    Everything that does not move -- the terrain, its contours, the lattice,
    the axis rails, the graph -- is painted once into an offscreen image and
    blitted. It is thrown away when the window resizes or the palette changes,
    and at no other time.
*/
class AtlasMapComponent final : public juce::Component
{
public:
    explicit AtlasMapComponent (AtlasFlowProcessor&);

    void paint (juce::Graphics&) override;
    void resized() override { stale = true; }
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;

    /** Arm drawing: the next drag lays down a path instead of moving the
        pointer, which is the one heavier gesture the surface carries. */
    void setDrawArmed (bool armed) { drawArmed = armed; stroke.clear(); repaint(); }
    bool isDrawArmed() const { return drawArmed; }

    /** The palette changed, so the cached layer describes the wrong colours. */
    void invalidateStatic() { stale = true; repaint(); }

private:
    void writePosition (juce::Point<float> local);
    juce::Rectangle<float> plane() const;
    juce::Point<float> presetPoint (int preset, juce::Rectangle<float> area) const;
    juce::Point<float> normalise (juce::Point<float> local) const;
    juce::Point<float> toScreen (juce::Point<float> normal, juce::Rectangle<float> area) const;
    int cellUnder (juce::Point<float> local, float radius = 13.0f) const;
    void rebuildCache (juce::Rectangle<float> area);
    void paintInfoCard (juce::Graphics&, const Palette&, int cell);
    void paintInset (juce::Graphics&, const Palette&, juce::Rectangle<float> area, int preset);

    const Palette& theme() const;
    int language() const;

    AtlasFlowProcessor& processor;
    AtlasGeometry geometry;

    /** What a move between two presets costs, which is not one question but
        three: is there a direct road, is there any road at all, or is the pair
        in two different colonies and therefore always a jump. */
    enum class Link
    {
        road,       // a mutual-kNN edge: the morph routes straight down it
        reachable,  // no direct road, but the graph connects them
        jump        // different components -- the voice restarts, every time
    };

    /** One edge of the drawn graph, with the three things worth saying about
        it: how close the pair really is, how much of that closeness survives
        the projection onto the two axes the picture has, and what a move
        across it actually does. */
    struct Edge { int a, b; float weight, shown; Link link; };
    std::vector<Edge> edges;

    // The ground is a 128x128 image stretched over the plane, which is what it
    // is: a smooth scalar field, blurry by nature. The contours are paths, so
    // the hairlines over it stay sharp at any scale factor. Both are rebuilt
    // when the plane moves or the palette changes, and at no other time.
    juce::Image ground;
    std::array<juce::Path, Terrain::kContours> contours;
    juce::Rectangle<float> cachedPlane;
    bool stale = true;
    int cachedTheme = -1;

    bool drawArmed = false, drawing = false;
    int hovered = -1;
    std::vector<juce::Point<float>> stroke;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AtlasMapComponent)
};

/** Three views of the same signal, stacked.

    The desktop instrument puts the scope in a drawer beside the map and lets
    each of the three panes show any of five views; so does this. One analyser
    pass per frame feeds all three, so the cost is the drawing, not the
    analysis.
*/
class ScopePaneComponent final : public juce::Component
{
public:
    explicit ScopePaneComponent (AtlasFlowProcessor&);

    void paint (juce::Graphics&) override;

    /** Pull a window from the tap and advance every pane's history. Separate
        from paint() because a paint can be triggered by the host for its own
        reasons, and the scroll rate must follow the clock rather than that. */
    void advance();

    void setKind (int slot, ScopeKind kind);
    ScopeKind kindFor (int slot) const { return views[(size_t) slot].getKind(); }
    void invalidateStatic() { for (auto& view : views) view.reset(); }

private:
    const Palette& theme() const;
    int language() const;

    AtlasFlowProcessor& processor;
    Analyser analyser;
    std::array<ScopeView, 3> views;
    double lastFrameMs = 0.0;
    float elapsed = 0.0f;
    long long seen = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ScopePaneComponent)
};

/** Everything that is not the map, behind one gear.

    Appearance and pane assignment only. The musical controls are parameters
    and belong to the host's own generic editor, where a producer can automate
    them; duplicating them here would invite the two to disagree.
*/
class SettingsPanel final : public juce::Component
{
public:
    SettingsPanel (AtlasFlowProcessor&, AtlasMapComponent&, ScopePaneComponent&,
                   std::function<void()> onAppearanceChange);

    void paint (juce::Graphics&) override;
    void resized() override;

    /** Re-read every label in the current language. Called when the language
        changes, which is the one setting that rewrites the panel it lives in. */
    void relabel();

private:
    const Palette& theme() const;
    int language() const;
    void styleBox (juce::ComboBox&);

    AtlasFlowProcessor& processor;
    AtlasMapComponent& mapView;
    ScopePaneComponent& scopePane;
    std::function<void()> appearanceChanged;

    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    juce::ComboBox languageBox, themeBox, voicesBox;
    std::array<juce::ComboBox, 3> slotBoxes;

    // Level, Morph, Lap, Temperature, Seed -- the parameters a player reaches
    // for. Attached to the APVTS rather than read from it, so moving one here
    // and automating it in the host are the same act, in the same undo history.
    std::array<juce::Slider, 5> sliders;
    std::vector<std::unique_ptr<SliderAttachment>> sliderLinks;
    std::unique_ptr<ComboAttachment> voicesLink;

    std::array<juce::Label, 11> captions;
    std::array<juce::Label, 3> headings;
    int proseTop = 0;          // where the rows stopped, so the prose follows them

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SettingsPanel)
};

class AtlasFlowEditor final : public juce::AudioProcessorEditor,
                              private juce::Timer
{
public:
    explicit AtlasFlowEditor (AtlasFlowProcessor&);
    ~AtlasFlowEditor() override = default;

    void paint (juce::Graphics&) override;
    void resized() override;

    /** One frame: pull a window into the analyser and advance every animation.

        The 30 Hz timer calls this in a host. It is public because the offline
        snapshot in tools/render_test.cpp has no message loop to run the timer
        -- runDispatchLoopUntil is gated behind JUCE_MODAL_LOOPS_PERMITTED,
        which a plugin build does not set -- and a picture of the scopes before
        they have seen any audio is not worth taking.
    */
    void tick();

    /** Open or close the settings panel, as the gear does.

        Public for the offline snapshot: a panel that is hidden by default is a
        panel no picture ever shows, and it is where every control that is not
        the map now lives. */
    void showSettings (bool shouldShow);

private:
    void timerCallback() override { tick(); }
    void applyAppearance();
    const Palette& theme() const;
    int language() const;

    AtlasFlowProcessor& processor;
    AtlasMapComponent mapView;
    ScopePaneComponent scopePane;
    SettingsPanel settings;
    juce::TextButton drawButton { "DRAW" }, clearButton { "CLEAR" },
                     gearButton { "SETTINGS" }, drawerButton { ">" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AtlasFlowEditor)
};

}  // namespace atlasflow

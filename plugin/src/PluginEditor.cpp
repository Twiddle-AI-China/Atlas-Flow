#include <algorithm>
#include <numeric>

#include "PluginEditor.h"

namespace atlasflow {

namespace {
constexpr float kMargin = 32.0f;        // keeps edge presets off the window border
constexpr int   kStatusHeight = 30;
constexpr int   kScopeHeadHeight = 20;
constexpr int   kSettingsWidth = 288;

/** Non-ASCII has to go through CharPointer_UTF8: a bare const char* is read in
    the platform's default encoding, which turns a middot into "Â·". */
const juce::String dot = juce::String::fromUTF8 (" \xc2\xb7 ");
const juce::String enDash = juce::String::fromUTF8 ("\xe2\x80\x93");

juce::String shortLabel (int preset)
{
    // serum_s048066 -> s048066. The prefix is the same fifty times over.
    return juce::String (Atlas::presetId (preset)).fromFirstOccurrenceOf ("_", false, false);
}
}

// ===========================================================================
// the map
// ===========================================================================

AtlasMapComponent::AtlasMapComponent (AtlasFlowProcessor& p) : processor (p)
{
    /*  The graph, measured once -- and it is two graphs, not one.

        The desktop instrument draws a k=3 nearest-neighbour graph: for every
        preset, the three nearest in the full 8-D space, unioned. That is 117
        edges and it makes the atlas look like one connected object.

        The planner does not route on that. It routes on the atlas's own
        *mutual* kNN -- 48 edges over 14 connected components -- and it asks
        one question of a move: are these two in the same component? If they
        are, graphPath finds a way and the timbre glides. If they are not,
        there is no path at all and the voice restarts.

        That splits the 121 drawn pairs three ways: 48 are a direct road, 19
        are near with no direct road but a route through the graph, and 54 are
        near and in a *different colony*, so a move across one is a jump every
        single time. The demo's caption gestures at this -- "that is usually
        where roaming jumps" -- but every line in its picture looks alike, so
        it cannot say which.

        Hue carries it, the way the palettes are built: the edge hue for a
        road, neutral ink for merely-reachable scenery, and the loner hue --
        the colour already used for a preset stranded on its own -- for the
        pairs that are cut off from each other.
    */
    std::array<std::array<float, kAtlasDims>, kPresetCount> unit {};
    for (int p2 = 0; p2 < kPresetCount; ++p2)
    {
        std::array<float, kAtlasDims> raw {};
        for (int d = 0; d < kAtlasDims; ++d)
            raw[(size_t) d] = kCoordinates[p2][d];
        unit[(size_t) p2] = geometry.normalize (raw);
    }
    // The demo ranks neighbours in plain normalised units, with no per-axis
    // scale, so this does too -- otherwise it would draw a different graph and
    // call it the same one. (The two metrics correlate at 0.986 and both give
    // the documented median 0.36, so the dashes mean the same thing either way.)
    auto separation = [&unit] (int a, int b)
    {
        float sum = 0.0f;
        for (int d = 0; d < kAtlasDims; ++d)
        {
            const float delta = unit[(size_t) a][(size_t) d] - unit[(size_t) b][(size_t) d];
            sum += delta * delta;
        }
        return std::sqrt (sum);
    };

    constexpr int kNearestDrawn = 3;
    std::array<std::array<bool, kPresetCount>, kPresetCount> drawn {};
    for (int a = 0; a < kPresetCount; ++a)
    {
        std::array<int, kPresetCount> order {};
        std::iota (order.begin(), order.end(), 0);
        std::partial_sort (order.begin(), order.begin() + kNearestDrawn + 1, order.end(),
                           [&] (int left, int right)
                           {
                               if (left == a) return true;
                               if (right == a) return false;
                               return separation (a, left) < separation (a, right);
                           });
        int taken = 0;
        for (int k = 0; k < kPresetCount && taken < kNearestDrawn; ++k)
        {
            const int b = order[(size_t) k];
            if (b == a)
                continue;
            drawn[(size_t) juce::jmin (a, b)][(size_t) juce::jmax (a, b)] = true;
            ++taken;
        }
    }
    // Every routable edge is drawn whether or not it is anyone's third-nearest:
    // four of them are not, and leaving those out would hide a route that
    // exists.
    for (int a = 0; a < kPresetCount; ++a)
        for (int b = a + 1; b < kPresetCount; ++b)
            if (Atlas::neighbours (a, b))
                drawn[(size_t) a][(size_t) b] = true;

    float nearest = 1.0e30f, farthest = 0.0f;
    for (int a = 0; a < kPresetCount; ++a)
        for (int b = a + 1; b < kPresetCount; ++b)
        {
            if (! drawn[(size_t) a][(size_t) b])
                continue;
            const float d = separation (a, b);
            const auto link = Atlas::neighbours (a, b) ? Link::road
                            : geometry.componentOf (a) == geometry.componentOf (b) ? Link::reachable
                            : Link::jump;
            // weight carries the raw distance until the whole range is known;
            // closeness is only meaningful relative to the other edges.
            edges.push_back ({ a, b, d, Atlas::foldedRatio (a, b), link });
            nearest = juce::jmin (nearest, d);
            farthest = juce::jmax (farthest, d);
        }
    const float span = juce::jmax (farthest - nearest, 1.0e-6f);
    for (auto& edge : edges)
        edge.weight = 1.0f - (edge.weight - nearest) / span;    // now closeness, 0..1

    setMouseCursor (juce::MouseCursor::CrosshairCursor);
}

const Palette& AtlasMapComponent::theme() const
{
    return palette (processor.appearance (appearanceKeys::theme, kMono));
}

int AtlasMapComponent::language() const
{
    return processor.appearance (appearanceKeys::language, systemLanguage());
}

juce::Rectangle<float> AtlasMapComponent::plane() const
{
    return getLocalBounds().toFloat().reduced (kMargin);
}

juce::Point<float> AtlasMapComponent::toScreen (juce::Point<float> normal,
                                                juce::Rectangle<float> area) const
{
    return { area.getX() + normal.x * area.getWidth(),
             area.getY() + normal.y * area.getHeight() };
}

juce::Point<float> AtlasMapComponent::presetPoint (int preset, juce::Rectangle<float> area) const
{
    float nx = 0.0f, ny = 0.0f;
    processor.atlas().displayPosition (preset, nx, ny);
    return toScreen ({ nx, ny }, area);
}

juce::Point<float> AtlasMapComponent::normalise (juce::Point<float> local) const
{
    const auto area = plane();
    if (area.getWidth() <= 0.0f || area.getHeight() <= 0.0f)
        return { 0.5f, 0.5f };
    return { juce::jlimit (0.0f, 1.0f, (local.x - area.getX()) / area.getWidth()),
             juce::jlimit (0.0f, 1.0f, (local.y - area.getY()) / area.getHeight()) };
}

void AtlasMapComponent::writePosition (juce::Point<float> local)
{
    const auto area = plane();
    if (area.getWidth() <= 0.0f || area.getHeight() <= 0.0f)
        return;
    const auto normal = normalise (local);

    // Go through the parameters rather than a side channel, so a drag is
    // indistinguishable from host automation and shows up in the host's undo
    // history and automation write.
    auto set = [this] (const char* id, float value)
    {
        if (auto* p = processor.state().getParameter (id))
        {
            p->beginChangeGesture();
            p->setValueNotifyingHost (value);
            p->endChangeGesture();
        }
    };
    set ("atlasX", normal.x);
    set ("atlasY", normal.y);
}

int AtlasMapComponent::cellUnder (juce::Point<float> local, float radius) const
{
    const auto area = plane();
    int best = -1;
    float bestDistance = radius;
    for (int p = 0; p < kPresetCount; ++p)
    {
        const float d = presetPoint (p, area).getDistanceFrom (local);
        if (d < bestDistance) { bestDistance = d; best = p; }
    }
    return best;
}

void AtlasMapComponent::mouseMove (const juce::MouseEvent& e)
{
    // Hover names a preset; it does not play one. In the desktop instrument
    // moving the pointer *is* the audition, which is the right gesture for a
    // page you opened to explore. In a host it would write two automation
    // parameters every time the mouse crossed the window, so here the pointer
    // has to be pressed to mean anything.
    const int under = cellUnder (e.position);
    if (under != hovered)
    {
        hovered = under;
        repaint();
    }
}

void AtlasMapComponent::mouseExit (const juce::MouseEvent&)
{
    if (hovered >= 0)
    {
        hovered = -1;
        repaint();
    }
}

void AtlasMapComponent::mouseDown (const juce::MouseEvent& e)
{
    if (drawArmed)
    {
        drawing = true;
        stroke.clear();
        stroke.push_back (normalise (e.position));
        repaint();
        return;
    }
    writePosition (e.position);
}

void AtlasMapComponent::mouseDrag (const juce::MouseEvent& e)
{
    if (drawing)
    {
        // Thin the stroke: raw pointer samples are far denser than the path
        // needs, and the speed profile is computed over its vertices.
        const auto here = normalise (e.position);
        if (stroke.empty() || here.getDistanceFrom (stroke.back()) > 0.02f)
        {
            stroke.push_back (here);
            repaint();
        }
        return;
    }
    writePosition (e.position);
}

void AtlasMapComponent::mouseUp (const juce::MouseEvent& e)
{
    if (! drawing)
    {
        writePosition (e.position);
        return;
    }
    drawing = false;
    if (stroke.size() >= 2)
        processor.setPath (stroke);
    else
        stroke.clear();
    repaint();
}

void AtlasMapComponent::rebuildCache (juce::Rectangle<float> area)
{
    const auto& colours = theme();
    ground = Terrain::shared().shaded (colours);
    contours = Terrain::shared().contourPaths (area);
    cachedPlane = area;
    cachedTheme = processor.appearance (appearanceKeys::theme, kMono);
    stale = false;
}

void AtlasMapComponent::paint (juce::Graphics& g)
{
    const auto& colours = theme();
    const auto area = plane();
    g.fillAll (colours.background);

    if (area.getWidth() < 8.0f || area.getHeight() < 8.0f)
        return;

    if (stale || area != cachedPlane
        || cachedTheme != processor.appearance (appearanceKeys::theme, kMono))
        rebuildCache (area);

    // --- the ground -------------------------------------------------------
    //
    // How much the runtime will overrule you. Bright is what-you-see-is-what-
    // you-hear; the dark basins are where your coordinate gets pulled into
    // legal territory before it reaches the model.
    //
    // It is drawn over the plane the presets occupy rather than over the whole
    // component. The desktop build stretches the same field across its entire
    // canvas while its presets sit inside a 64 px pad, so there the terrain is
    // offset from the atlas it describes by the width of that pad.
    g.setImageResamplingQuality (juce::Graphics::highResamplingQuality);
    g.drawImage (ground, area, juce::RectanglePlacement::stretchToFit);

    for (int index = 0; index < Terrain::kContours; ++index)
    {
        // Higher ground gets a firmer line, so the rings read as a stack
        // rather than as a single tangle at one weight.
        g.setColour (colours.tint (0.05f + 0.055f * (float) index));
        g.strokePath (contours[(size_t) index], juce::PathStrokeType (1.0f));
    }

    // A sparse lattice so empty atlas regions read as space rather than void.
    g.setColour (colours.tint (0.055f));
    for (float x = std::fmod (area.getWidth() * 0.5f, 42.0f); x < area.getWidth(); x += 42.0f)
        for (float y = std::fmod (area.getHeight() * 0.5f, 42.0f); y < area.getHeight(); y += 42.0f)
            g.fillRect (std::round (area.getX() + x), std::round (area.getY() + y), 1.0f, 1.0f);

    // Axis rails: the map is a projection with units, so label them.
    g.setFont (juce::FontOptions (9.0f));
    for (int step = -4; step <= 4; ++step)
    {
        const float value = (float) step / 4.0f;
        const auto at = toScreen ({ (value + 1.0f) * 0.5f, 1.0f - (value + 1.0f) * 0.5f }, area);
        const float length = step % 2 ? 4.0f : 8.0f;
        g.setColour (colours.tint (0.14f));
        g.drawLine (area.getX() - length, std::round (at.y) + 0.5f, area.getX(),
                    std::round (at.y) + 0.5f, 1.0f);
        g.drawLine (std::round (at.x) + 0.5f, area.getBottom(), std::round (at.x) + 0.5f,
                    area.getBottom() + length, 1.0f);
        if (step % 2 == 0)
        {
            g.setColour (colours.tint (0.30f));
            // The corner belongs to one rail or the other, not both: the two
            // -1.0 labels land on the same pixels and print as "-1.01.0".
            if (at.y < area.getBottom() - 40.0f)
                g.drawText (juce::String (value, 1), (int) area.getX() + 4, (int) at.y - 11, 30, 10,
                            juce::Justification::left, false);
            g.drawText (juce::String (value, 1), (int) at.x - 15, (int) area.getBottom() + 3, 30, 10,
                        juce::Justification::centred, false);
        }
    }

    // --- the graph --------------------------------------------------------
    //
    // Who is next to whom, measured in the full 8-D space rather than in
    // screen distance. Dashed marks the worst quarter: adjacent on screen, far
    // apart in fact, and that is usually where roaming jumps.
    const int audible = processor.audiblePreset.load (std::memory_order_relaxed);
    for (const auto& edge : edges)
    {
        const juce::Line<float> line { presetPoint (edge.a, area), presetPoint (edge.b, area) };
        const bool lit = (edge.a == audible || edge.b == audible);

        if (edge.link != Link::road)
        {
            const float dots[] = { 1.0f, 3.0f };
            if (edge.link == Link::jump)
            {
                // Two colonies. However near these look, a move between them
                // has no path to follow and the voice starts over -- so they
                // get the colour a stranded preset gets, for the same reason.
                g.setColour (colours.loner.withAlpha (lit ? 0.75f : 0.30f + 0.25f * edge.weight));
            }
            else
            {
                // Near, no direct road, but the graph gets there. Scenery.
                g.setColour (colours.tint (lit ? 0.13f : 0.05f + 0.03f * edge.weight));
            }
            g.drawDashedLine (line, dots, 2, 1.0f);
            continue;
        }

        const float alpha = lit ? 0.38f + 0.42f * edge.weight : 0.10f + 0.22f * edge.weight;
        g.setColour (colours.edge.withAlpha (alpha));
        if (edge.shown < Atlas::kFoldedThreshold)
        {
            const float dashes[] = { 2.0f, 4.0f };
            g.drawDashedLine (line, dashes, 2, 1.0f);
        }
        else
        {
            g.drawLine (line, 1.0f);
        }
    }

    // --- the presets ------------------------------------------------------
    for (int p = 0; p < kPresetCount; ++p)
    {
        const auto at = presetPoint (p, area);
        const bool alone = geometry.components()[(size_t) geometry.componentOf (p)].size() == 1;

        if (Atlas::splitOf (p) != kTrain)
        {
            // Held out: no weight was ever fitted to this preset, so what you
            // hear standing on it is entirely generated. The demo rings the
            // three test presets; the two validation ones are ringed here too,
            // but dashed -- they trained no weights either, and they did steer
            // checkpoint selection, so neither "seen" nor "unseen" is the whole
            // truth and the dash is where the difference goes.
            g.setColour (colours.test.withAlpha (0.70f));
            if (Atlas::splitOf (p) == kTest)
            {
                g.drawEllipse (at.x - 11.0f, at.y - 11.0f, 22.0f, 22.0f, 1.0f);
            }
            else
            {
                const float dashes[] = { 2.0f, 3.0f };
                juce::Path ring, dashed;
                ring.addEllipse (at.x - 11.0f, at.y - 11.0f, 22.0f, 22.0f);
                juce::PathStrokeType (1.0f).createDashedStroke (dashed, ring, dashes, 2);
                g.fillPath (dashed);
            }
        }

        const float radius = p == hovered ? 6.0f : alone ? 3.0f : 4.2f;
        g.setColour (p == hovered ? colours.ink : alone ? colours.loner : colours.cell);
        g.fillEllipse (at.x - radius, at.y - radius, radius * 2.0f, radius * 2.0f);
    }

    // A double ring says "this preset is what you are hearing", which a plain
    // dot among fifty dots cannot.
    {
        const auto at = presetPoint (audible, area);
        g.setColour (colours.accent.withAlpha (0.85f));
        for (const float radius : { 13.0f, 17.0f })
            g.drawEllipse (at.x - radius, at.y - radius, radius * 2.0f, radius * 2.0f, 1.0f);
    }

    // --- the drawn path ---------------------------------------------------
    {
        const auto points = drawing ? stroke : processor.pathPoints();
        if (points.size() >= 2)
        {
            juce::Path trace;
            trace.startNewSubPath (toScreen (points.front(), area));
            for (size_t i = 1; i < points.size(); ++i)
                trace.lineTo (toScreen (points[i], area));

            g.setColour (colours.path.withAlpha (drawing ? 0.55f : 0.85f));
            g.strokePath (trace, juce::PathStrokeType (1.2f));

            for (auto end : { points.front(), points.back() })
            {
                const auto at = toScreen (end, area);
                g.drawRect (juce::Rectangle<float> { at.x - 3.0f, at.y - 3.0f, 6.0f, 6.0f }, 1.0f);
            }

            if (! drawing && processor.pathActive())
            {
                // The head glides; the voice marker below steps along behind
                // it, so the gap between them is how far the sound lags.
                PathData snapshot;
                snapshot.build (points);
                const auto head = toScreen (snapshot.at (processor.pathPhase()), area);
                g.setColour (colours.path);
                g.fillEllipse (head.x - 3.5f, head.y - 3.5f, 7.0f, 7.0f);
                g.setColour (colours.path.withAlpha (0.4f));
                g.drawEllipse (head.x - 8.0f, head.y - 8.0f, 16.0f, 16.0f, 1.0f);
            }
        }
    }

    // --- pointer, voice, and the leash between them -----------------------
    const float px = processor.state().getRawParameterValue ("atlasX")->load();
    const float py = processor.state().getRawParameterValue ("atlasY")->load();
    const auto cursor = toScreen ({ juce::jlimit (0.0f, 1.0f, px),
                                    juce::jlimit (0.0f, 1.0f, py) }, area);

    const float vx = processor.voiceX.load (std::memory_order_relaxed);
    const float vy = processor.voiceY.load (std::memory_order_relaxed);
    const bool haveVoice = vx >= 0.0f && vy >= 0.0f;
    const auto voice = toScreen ({ vx, vy }, area);

    // The leash makes the lag explicit: the voice is still travelling to the
    // coordinate you are pointing at -- or it has been pulled off it, which on
    // a dark part of the terrain is permanent rather than temporary.
    if (haveVoice && voice.getDistanceFrom (cursor) > 14.0f)
    {
        const float dashes[] = { 2.0f, 3.0f };
        g.setColour (colours.accent.withAlpha (0.35f));
        g.drawDashedLine ({ cursor, voice }, dashes, 2, 1.0f);
    }

    if (haveVoice)
    {
        // A dashed reticle holds the same amount of attention as a soft halo
        // without blurring the hairlines it sits on top of.
        const float dashes[] = { 2.0f, 4.0f };
        g.setColour (colours.accent.withAlpha (0.45f));
        juce::Path ring;
        ring.addEllipse (voice.x - 17.0f, voice.y - 17.0f, 34.0f, 34.0f);
        juce::Path dashed;
        juce::PathStrokeType (1.0f).createDashedStroke (dashed, ring, dashes, 2);
        g.fillPath (dashed);

        g.setColour (colours.accent);
        g.drawEllipse (voice.x - 6.0f, voice.y - 6.0f, 12.0f, 12.0f, 1.5f);
        for (const auto& tick : { juce::Line<float> { voice.x - 11.0f, voice.y, voice.x - 8.0f, voice.y },
                                  juce::Line<float> { voice.x + 8.0f, voice.y, voice.x + 11.0f, voice.y },
                                  juce::Line<float> { voice.x, voice.y - 11.0f, voice.x, voice.y - 8.0f },
                                  juce::Line<float> { voice.x, voice.y + 8.0f, voice.x, voice.y + 11.0f } })
            g.drawLine (tick, 1.5f);
    }

    // Where the parameters point. A crosshair rather than a filled handle so
    // it never hides the preset underneath it.
    g.setColour (colours.accent2.withAlpha (0.75f));
    g.drawEllipse (cursor.x - 11.0f, cursor.y - 11.0f, 22.0f, 22.0f, 1.0f);
    g.drawLine (cursor.x - 9.0f, cursor.y, cursor.x + 9.0f, cursor.y, 1.0f);
    g.drawLine (cursor.x, cursor.y - 9.0f, cursor.x, cursor.y + 9.0f, 1.0f);

    // The inset is a luxury; on a narrow map it would cover the atlas it
    // annotates, so it only appears when there is room for it.
    if (area.getWidth() > 560.0f)
        paintInset (g, colours, area, audible);

    if (hovered >= 0)
    {
        const auto at = presetPoint (hovered, area);
        g.setColour (colours.ink);
        g.setFont (juce::FontOptions (10.0f, juce::Font::bold));
        g.drawText (shortLabel (hovered), (int) at.x - 50, (int) at.y - 27, 100, 12,
                    juce::Justification::centred, false);
        paintInfoCard (g, colours, hovered);
    }
}

/** All eight axes of the sounding sound, since the map can only draw two.

    PC1 and PC2 hold 42.3% and 19.8% of the variance, so most of what separates
    one pad from another happens on axes this picture cannot show. The bars are
    the nearest preset's own coordinate; the ticks over them are where the
    voice actually is, which is the same override the terrain shades, read off
    one point instead of the whole plane.
*/
void AtlasMapComponent::paintInset (juce::Graphics& g, const Palette& colours,
                                    juce::Rectangle<float> area, int preset)
{
    constexpr float width = 132.0f, height = 92.0f;
    const juce::Rectangle<float> card { area.getRight() - width - 18.0f, area.getY() + 18.0f,
                                        width, height };

    const auto ring = presetPoint (preset, area);
    const juce::Point<float> anchor { card.getX(), card.getCentreY() };
    if (ring.getDistanceFrom (anchor) > 60.0f)
    {
        // Mostly orthogonal, like an annotation rule rather than a drawn line,
        // and faint enough that it does not compete with the cells it crosses.
        const float elbow = anchor.x - 26.0f;
        juce::Path rule;
        rule.startNewSubPath (ring.x + 19.0f, ring.y);
        rule.lineTo (elbow, ring.y);
        rule.lineTo (elbow, anchor.y);
        rule.lineTo (anchor.x, anchor.y);
        const float dashes[] = { 3.0f, 4.0f };
        juce::Path dashed;
        juce::PathStrokeType (1.0f).createDashedStroke (dashed, rule, dashes, 2);
        g.setColour (colours.tint (0.13f));
        g.fillPath (dashed);
    }

    g.setColour (colours.shade (0.72f));
    g.fillRect (card);
    g.setColour (colours.tint (0.28f));
    g.drawRect (card, 1.0f);

    g.setFont (juce::FontOptions (9.0f));
    g.setColour (colours.tint (0.55f));
    g.drawText ("[8D] " + shortLabel (preset),
                juce::Rectangle<float> { card.getX() + 7.0f, card.getY() + 4.0f, width - 14.0f, 11.0f },
                juce::Justification::left, false);

    std::array<float, kAtlasDims> here {};
    processor.atlas().normalisedCoordinate (kCoordinates[preset], here.data());

    const float usable = width - 34.0f;
    const float middle = card.getX() + 20.0f + usable * 0.5f;
    for (int axis = 0; axis < kAtlasDims; ++axis)
    {
        const float y = card.getY() + 24.0f + (float) axis * 8.0f;
        g.setColour (colours.tint (0.34f));
        g.drawText (juce::String (axis + 1),
                    juce::Rectangle<float> { card.getX() + 5.0f, y - 4.0f, 12.0f, 9.0f },
                    juce::Justification::left, false);
        g.setColour (colours.tint (0.14f));
        g.fillRect (card.getX() + 20.0f, y, usable, 1.0f);              // the zero line

        const float extent = juce::jlimit (-1.0f, 1.0f, here[(size_t) axis]) * usable * 0.5f;
        g.setColour (colours.tint (0.85f));
        g.fillRect (juce::jmin (middle, middle + extent), y - 2.0f, std::abs (extent), 4.0f);
        g.setColour (colours.tint (0.30f));
        g.fillRect (std::round (middle), y - 4.0f, 1.0f, 8.0f);         // centre tick

        if (processor.voiceX.load (std::memory_order_relaxed) >= 0.0f)
        {
            const float live = juce::jlimit (-1.0f, 1.0f,
                processor.voiceCoordinate[(size_t) axis].load (std::memory_order_relaxed));
            g.setColour (colours.accent.withAlpha (0.9f));
            g.fillRect (std::round (middle + live * usable * 0.5f), y - 3.0f, 1.0f, 7.0f);
        }
    }
}

/** What the map knows about one preset, in the corner, while you hover it. */
void AtlasMapComponent::paintInfoCard (juce::Graphics& g, const Palette& colours, int cell)
{
    const auto& group = geometry.components()[(size_t) geometry.componentOf (cell)];

    // The nearest neighbour, and how much of that closeness the picture can
    // actually show. A low figure means this dot's neighbours are drawn far
    // nearer than they are.
    int nearest = -1;
    float nearestDistance = 1.0e30f;
    for (int p = 0; p < kPresetCount; ++p)
    {
        if (p == cell)
            continue;
        const float d = AtlasGeometry::distance (kCoordinates[cell], kCoordinates[p]);
        if (d < nearestDistance) { nearestDistance = d; nearest = p; }
    }

    const int lang = language();
    const juce::String split =
        Atlas::splitOf (cell) == kTest ? tr (text::heldTest, lang)
      : Atlas::splitOf (cell) == kValidation ? tr (text::heldValid, lang)
      : tr (text::trained, lang);
    const juce::String body =
        split + dot + (group.size() == 1 ? tr (text::alone, lang)
                                         : tr (text::colonyOf, lang) + juce::String ((int) group.size()));
    const juce::String meta =
        tr (text::nearest, lang) + shortLabel (nearest) + dot
        + juce::String (juce::roundToInt (Atlas::foldedRatio (cell, nearest) * 100.0f))
        + tr (text::drawable, lang);

    const auto card = juce::Rectangle<int> { 12, getHeight() - 60, 300, 48 };
    g.setColour (colours.shade (0.82f));
    g.fillRect (card);
    g.setColour (colours.line);
    g.drawRect (card, 1);

    auto text = card.reduced (8, 6);
    g.setColour (colours.ink);
    g.setFont (juce::FontOptions (10.5f, juce::Font::bold));
    g.drawText (shortLabel (cell), text.removeFromTop (13), juce::Justification::left, false);
    g.setColour (colours.dim);
    g.setFont (juce::FontOptions (9.5f));
    g.drawText (body, text.removeFromTop (12), juce::Justification::left, false);
    g.setColour (colours.muted);
    g.drawText (meta, text.removeFromTop (12), juce::Justification::left, false);
}

// ===========================================================================
// the scope
// ===========================================================================

ScopePaneComponent::ScopePaneComponent (AtlasFlowProcessor& p) : processor (p)
{
    lastFrameMs = juce::Time::getMillisecondCounterHiRes();
    views[0].setKind ((ScopeKind) processor.appearance (appearanceKeys::slot0, 0));
    views[1].setKind ((ScopeKind) processor.appearance (appearanceKeys::slot1, 1));
    views[2].setKind ((ScopeKind) processor.appearance (appearanceKeys::slot2, 2));
}

const Palette& ScopePaneComponent::theme() const
{
    return palette (processor.appearance (appearanceKeys::theme, kMono));
}

int ScopePaneComponent::language() const
{
    return processor.appearance (appearanceKeys::language, systemLanguage());
}

void ScopePaneComponent::setKind (int slot, ScopeKind kind)
{
    views[(size_t) slot].setKind (kind);
    static const juce::Identifier* keys[] { &appearanceKeys::slot0, &appearanceKeys::slot1,
                                           &appearanceKeys::slot2 };
    processor.setAppearance (*keys[slot], (int) kind);
    repaint();
}

void ScopePaneComponent::advance()
{
    const double now = juce::Time::getMillisecondCounterHiRes();
    // Clamped: a stalled message thread would otherwise scroll a whole screen
    // of history past in one frame, which looks like a glitch rather than like
    // the pause it actually was.
    elapsed = (float) juce::jlimit (0.0, 0.2, (now - lastFrameMs) / 1000.0);
    lastFrameMs = now;
    analyser.update (processor.scope);
    seen = processor.scope.samplesSeen();
}

void ScopePaneComponent::paint (juce::Graphics& g)
{
    const auto& colours = theme();
    g.fillAll (colours.background);

    auto bounds = getLocalBounds();
    auto head = bounds.removeFromTop (kScopeHeadHeight);

    const double rate = processor.getSampleRate() > 0.0 ? processor.getSampleRate() : 44100.0;
    g.setColour (colours.dim);
    g.setFont (juce::FontOptions (9.0f));
    g.drawText ("[SCOPE]", head.reduced (6, 0), juce::Justification::left, false);
    g.setColour (colours.muted);
    g.drawText ("40Hz" + enDash + juce::String (scopeTop (rate) / 1000.0f, 1) + "kHz" + dot
                    + tr (text::stream, language()) + " " + juce::String (rate / 2000.0, 1) + "kHz",
                head.reduced (6, 0), juce::Justification::right, false);

    g.setColour (colours.line);
    g.drawLine ((float) bounds.getX(), (float) bounds.getY(),
                (float) bounds.getRight(), (float) bounds.getY(), 1.0f);

    const int cellHeight = bounds.getHeight() / 3;
    const bool dropped = false;    // nothing can drop: there is no stream here
    for (int slot = 0; slot < 3; ++slot)
    {
        auto cell = slot == 2 ? bounds : bounds.removeFromTop (cellHeight);
        auto frame = cell.reduced (1);
        views[(size_t) slot].paint (g, frame, analyser, colours, elapsed, rate, dropped, language());

        g.setColour (colours.muted);
        g.setFont (juce::FontOptions (9.0f));
        g.drawText (scopeName (views[(size_t) slot].getKind()),
                    frame.getX() + 6, frame.getY() + 3, 140, 10, juce::Justification::left, false);
        g.setColour (colours.line);
        g.drawLine ((float) cell.getX(), (float) cell.getBottom(),
                    (float) cell.getRight(), (float) cell.getBottom(), 1.0f);
    }
}

// ===========================================================================
// settings
// ===========================================================================

SettingsPanel::SettingsPanel (AtlasFlowProcessor& p, AtlasMapComponent& m, ScopePaneComponent& s,
                              std::function<void()> onAppearanceChange)
    : processor (p), mapView (m), scopePane (s), appearanceChanged (std::move (onAppearanceChange))
{
    for (auto& caption : captions)
    {
        caption.setFont (juce::FontOptions (9.5f));
        addAndMakeVisible (caption);
    }
    for (auto& heading : headings)
    {
        heading.setFont (juce::FontOptions (9.0f, juce::Font::bold));
        addAndMakeVisible (heading);
    }

    // The language names itself in itself: a menu that said "Chinese" in
    // English would be unreadable to exactly the person reaching for it.
    languageBox.addItem ("English", kEnglish + 1);
    languageBox.addItem (juce::String::fromUTF8 ("\xe4\xb8\xad\xe6\x96\x87"), kChinese + 1);
    languageBox.setSelectedId (processor.appearance (appearanceKeys::language, systemLanguage()) + 1,
                               juce::dontSendNotification);
    languageBox.onChange = [this]
    {
        processor.setAppearance (appearanceKeys::language, languageBox.getSelectedId() - 1);
        relabel();
        if (appearanceChanged)
            appearanceChanged();
    };
    styleBox (languageBox);
    addAndMakeVisible (languageBox);

    for (int i = 0; i < kPaletteCount; ++i)
        themeBox.addItem (palette (i).label, i + 1);
    themeBox.setSelectedId (processor.appearance (appearanceKeys::theme, kMono) + 1,
                            juce::dontSendNotification);
    themeBox.onChange = [this]
    {
        processor.setAppearance (appearanceKeys::theme, themeBox.getSelectedId() - 1);
        // The map's cached ground and the scopes' painted histories are both in
        // the old palette, so both have to be thrown away rather than drawn
        // over -- the same reason applyTheme() clears them in the desktop build.
        mapView.invalidateStatic();
        scopePane.invalidateStatic();
        if (appearanceChanged)
            appearanceChanged();
    };
    styleBox (themeBox);
    addAndMakeVisible (themeBox);

    // The parameters a player reaches for. Voices is a choice because its
    // options are not interchangeable -- each one names the latency it costs.
    if (auto* voices = dynamic_cast<juce::AudioParameterChoice*> (processor.state().getParameter ("voices")))
    {
        voicesBox.addItemList (voices->choices, 1);
        voicesLink = std::make_unique<ComboAttachment> (processor.state(), "voices", voicesBox);
    }
    styleBox (voicesBox);
    addAndMakeVisible (voicesBox);

    static const char* parameterIds[] { "level", "morph", "lap", "temperature", "seed" };
    // The decimals and the unit come from the parameter itself (see layout()),
    // so this box only has to be wide enough to hold what the parameter says.
    for (size_t i = 0; i < sliders.size(); ++i)
    {
        auto& slider = sliders[i];
        slider.setSliderStyle (juce::Slider::LinearHorizontal);
        slider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 60, 18);
        addAndMakeVisible (slider);
        sliderLinks.push_back (std::make_unique<SliderAttachment> (
            processor.state(), parameterIds[i], slider));
    }

    for (int slot = 0; slot < 3; ++slot)
    {
        auto& box = slotBoxes[(size_t) slot];
        for (int kind = 0; kind < kScopeKindCount; ++kind)
            box.addItem (scopeName ((ScopeKind) kind), kind + 1);
        box.setSelectedId ((int) scopePane.kindFor (slot) + 1, juce::dontSendNotification);
        box.onChange = [this, slot]
        {
            scopePane.setKind (slot, (ScopeKind) (slotBoxes[(size_t) slot].getSelectedId() - 1));
        };
        styleBox (box);
        addAndMakeVisible (box);
    }

    relabel();
}

void SettingsPanel::relabel()
{
    const int lang = language();
    const Phrase* names[] { &text::language, &text::themeName,
                            &text::level, &text::morph, &text::lap, &text::voiceCount,
                            &text::temperature, &text::seed,
                            &text::topPane, &text::middlePane, &text::bottomPane };
    static_assert (std::size (names) == 11, "a caption lost its label");
    for (size_t i = 0; i < captions.size(); ++i)
        captions[i].setText (tr (*names[i], lang), juce::dontSendNotification);

    const Phrase* sections[] { &text::sectionLook, &text::sectionPlay, &text::sectionScope };
    for (size_t i = 0; i < headings.size(); ++i)
        headings[i].setText (tr (*sections[i], lang), juce::dontSendNotification);
    repaint();
}

int SettingsPanel::language() const
{
    return processor.appearance (appearanceKeys::language, systemLanguage());
}

void SettingsPanel::styleBox (juce::ComboBox& box)
{
    const auto& colours = theme();
    box.setColour (juce::ComboBox::backgroundColourId, colours.panel);
    box.setColour (juce::ComboBox::outlineColourId, colours.line);
    box.setColour (juce::ComboBox::textColourId, colours.ink);
    box.setColour (juce::ComboBox::arrowColourId, colours.dim);
}

const Palette& SettingsPanel::theme() const
{
    return palette (processor.appearance (appearanceKeys::theme, kMono));
}

void SettingsPanel::paint (juce::Graphics& g)
{
    const auto& colours = theme();
    g.fillAll (colours.panel);
    g.setColour (colours.line);
    g.drawLine (0.0f, 0.0f, 0.0f, (float) getHeight(), 1.0f);

    for (auto* box : { &languageBox, &themeBox, &voicesBox,
                       &slotBoxes[0], &slotBoxes[1], &slotBoxes[2] })
        styleBox (*box);
    for (auto& caption : captions)
        caption.setColour (juce::Label::textColourId, colours.dim);
    for (auto& heading : headings)
        heading.setColour (juce::Label::textColourId, colours.muted);
    for (auto& slider : sliders)
    {
        slider.setColour (juce::Slider::trackColourId, colours.ink.withAlpha (0.55f));
        slider.setColour (juce::Slider::backgroundColourId, colours.line);
        slider.setColour (juce::Slider::thumbColourId, colours.accent);
        slider.setColour (juce::Slider::textBoxTextColourId, colours.ink);
        slider.setColour (juce::Slider::textBoxBackgroundColourId, colours.panel);
        slider.setColour (juce::Slider::textBoxOutlineColourId, colours.line);
    }

    const int lang = language();
    g.setColour (colours.muted);
    g.setFont (juce::FontOptions (9.5f));
    g.drawFittedText (tr (text::aboutTheme, lang) + "\n\n" + tr (text::aboutScope, lang)
                          + "\n\n" + tr (text::aboutParameters, lang),
                      12, proseTop, getWidth() - 24, juce::jmax (0, getHeight() - proseTop - 12),
                      juce::Justification::topLeft, 24);
}

void SettingsPanel::resized()
{
    // Label on the left, control on the right, one row each -- the demo's
    // settings grid. Stacking captions above their controls instead needed
    // 600 px of height for eleven rows and left no room for anything else.
    auto bounds = getLocalBounds().reduced (12, 12);
    constexpr int kLabelWidth = 76;
    constexpr int kRowHeight = 23;

    auto heading = [&bounds] (juce::Label& label)
    {
        bounds.removeFromTop (8);
        label.setBounds (bounds.removeFromTop (14));
        bounds.removeFromTop (2);
    };
    auto row = [&bounds] (juce::Label& caption, juce::Component& control)
    {
        auto line = bounds.removeFromTop (kRowHeight);
        caption.setBounds (line.removeFromLeft (kLabelWidth));
        control.setBounds (line.reduced (0, 1));
        bounds.removeFromTop (3);
    };

    heading (headings[0]);
    row (captions[0], languageBox);
    row (captions[1], themeBox);

    heading (headings[1]);
    row (captions[2], sliders[0]);      // level
    row (captions[3], sliders[1]);      // morph
    row (captions[4], sliders[2]);      // lap
    row (captions[5], voicesBox);
    row (captions[6], sliders[3]);      // temperature
    row (captions[7], sliders[4]);      // seed

    heading (headings[2]);
    row (captions[8], slotBoxes[0]);
    row (captions[9], slotBoxes[1]);
    row (captions[10], slotBoxes[2]);

    proseTop = bounds.getY() + 10;
}

// ===========================================================================
// the window
// ===========================================================================

AtlasFlowEditor::AtlasFlowEditor (AtlasFlowProcessor& p)
    : AudioProcessorEditor (&p), processor (p), mapView (p), scopePane (p),
      settings (p, mapView, scopePane, [this] { applyAppearance(); })
{
    addAndMakeVisible (mapView);
    addAndMakeVisible (scopePane);
    addChildComponent (settings);

    drawButton.setClickingTogglesState (true);
    gearButton.setClickingTogglesState (true);
    drawerButton.setClickingTogglesState (true);
    drawerButton.setToggleState (processor.appearance (appearanceKeys::drawerShut, 0) != 0,
                                 juce::dontSendNotification);

    for (auto* b : { &drawButton, &clearButton, &gearButton, &drawerButton })
        addAndMakeVisible (b);

    drawButton.onClick = [this] { mapView.setDrawArmed (drawButton.getToggleState()); };
    clearButton.onClick = [this]
    {
        processor.clearPath();
        drawButton.setToggleState (false, juce::dontSendNotification);
        mapView.setDrawArmed (false);
    };
    gearButton.onClick = [this]
    {
        settings.setVisible (gearButton.getToggleState());
        resized();
    };
    drawerButton.onClick = [this]
    {
        processor.setAppearance (appearanceKeys::drawerShut, drawerButton.getToggleState() ? 1 : 0);
        resized();
    };

    applyAppearance();
    setResizable (true, true);
    setResizeLimits (560, 380, 2400, 1800);
    setSize (1040, 660);
    startTimerHz (30);
}

const Palette& AtlasFlowEditor::theme() const
{
    return palette (processor.appearance (appearanceKeys::theme, kMono));
}

int AtlasFlowEditor::language() const
{
    return processor.appearance (appearanceKeys::language, systemLanguage());
}

void AtlasFlowEditor::applyAppearance()
{
    const auto& colours = theme();
    const int lang = language();
    for (auto* b : { &drawButton, &clearButton, &gearButton, &drawerButton })
    {
        b->setColour (juce::TextButton::buttonColourId, colours.panel);
        b->setColour (juce::TextButton::buttonOnColourId, colours.ink);
        b->setColour (juce::TextButton::textColourOffId, colours.dim);
        b->setColour (juce::TextButton::textColourOnId, colours.background);
    }
    drawButton.setButtonText (tr (text::draw, lang));
    clearButton.setButtonText (tr (text::clear, lang));
    gearButton.setButtonText (tr (text::settings, lang));
    resized();
    repaint();
}

void AtlasFlowEditor::showSettings (bool shouldShow)
{
    gearButton.setToggleState (shouldShow, juce::sendNotificationSync);
}

void AtlasFlowEditor::tick()
{
    scopePane.advance();
    scopePane.repaint();
    mapView.repaint();
    repaint (getLocalBounds().removeFromBottom (kStatusHeight));
}

void AtlasFlowEditor::resized()
{
    auto bounds = getLocalBounds();
    bounds.removeFromBottom (kStatusHeight);

    if (settings.isVisible())
        settings.setBounds (bounds.removeFromRight (juce::jmin (kSettingsWidth, bounds.getWidth() / 2)));

    if (! drawerButton.getToggleState())
    {
        // The scope earns its width on a big display and must not eat the map
        // on a small one, so it scales with the window between hard bounds.
        const int width = juce::jlimit (200, 430, juce::roundToInt (bounds.getWidth() * 0.26));
        scopePane.setVisible (true);
        scopePane.setBounds (bounds.removeFromRight (juce::jmin (width, bounds.getWidth() / 2)));
    }
    else
    {
        scopePane.setVisible (false);
    }

    mapView.setBounds (bounds);

    auto strip = getLocalBounds().removeFromBottom (kStatusHeight);
    strip.removeFromRight (8);
    gearButton.setBounds (strip.removeFromRight (74).reduced (2, 4));
    clearButton.setBounds (strip.removeFromRight (56).reduced (2, 4));
    drawButton.setBounds (strip.removeFromRight (52).reduced (2, 4));
    drawerButton.setBounds (strip.removeFromRight (26).reduced (2, 4));
    drawerButton.setButtonText (drawerButton.getToggleState() ? "<" : ">");
}

void AtlasFlowEditor::paint (juce::Graphics& g)
{
    const auto& colours = theme();
    g.fillAll (colours.background);

    auto strip = getLocalBounds().removeFromBottom (kStatusHeight).reduced (10, 0);
    // Leave the right-hand end to the buttons, or the readout runs under them.
    auto textArea = strip.withTrimmedRight (220);
    g.setColour (colours.line);
    g.drawLine ((float) strip.getX(), (float) strip.getY(),
                (float) strip.getRight(), (float) strip.getY(), 1.0f);

    const int preset = processor.audiblePreset.load (std::memory_order_relaxed);
    const int voices = processor.activeVoices.load (std::memory_order_relaxed);
    const float s = processor.audibleSine.load (std::memory_order_relaxed);
    const float w = processor.audibleSaw.load (std::memory_order_relaxed);
    const float q = processor.audibleSquare.load (std::memory_order_relaxed);

    const int lang = language();
    juce::String readout =
        "[" + processor.engineDescription() + "] " + Atlas::presetId (preset)
        + "   " + tr (text::voices, lang) + " " + juce::String (voices)
        + "/" + juce::String (AtlasFlowProcessor::kMaxVoices);
    if (! processor.modelLoaded())
        readout += "   sine " + juce::String (juce::roundToInt (s * 100.0f)) + "%"
                 + dot + "saw " + juce::String (juce::roundToInt (w * 100.0f)) + "%"
                 + dot + "square " + juce::String (juce::roundToInt (q * 100.0f)) + "%";
    else
        readout += "   " + tr (text::bank, lang) + " " + juce::String (processor.bankReady()) + "/36";

    g.setColour (colours.dim);
    g.setFont (juce::FontOptions (12.0f));
    g.drawText (readout, textArea, juce::Justification::centredLeft, false);

    juce::String right = juce::String ("MIDI 36-71") + dot + tr (text::folded, lang);
    if (processor.pathActive())
        right = tr (text::path, lang) + " "
              + juce::String (juce::roundToInt (processor.pathPhase() * 100.0f)) + "%" + dot + right;
    if (processor.modelLoaded() && processor.planning.load (std::memory_order_relaxed))
        right = tr (text::planning, lang) + dot + right;

    g.setColour (colours.muted);
    g.drawText (right, textArea, juce::Justification::centredRight, false);
}

}  // namespace atlasflow

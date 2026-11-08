#pragma once

#include <array>

#include <juce_graphics/juce_graphics.h>

namespace atlasflow {

/*  The five palettes, ported from atlas-flow-web-demo/styles.css.

    A theme changes the palette and nothing else -- same hairlines, same
    layout, same information. Structure (the lattice, the axis rails, the
    labels, the frames) stays neutral in every one of them and is drawn with
    tint()/shade(); only the things that *mean* something get a hue of their
    own: which preset is sounding, where you are pointing, the path you drew,
    which cells the model never saw.

    Monochrome is still the default. A plugin window sits inside someone else's
    colour scheme all day, and greyscale is the one choice that never fights
    it -- but the desktop instrument offers five, so the plugin offers five.
*/

struct Palette
{
    const char* key;
    const char* label;

    juce::Colour background, panel, line, lineSoft, ink, dim, muted;

    // The roles. Named, not indexed, because the point of them is that each
    // one says a particular thing.
    juce::Colour cell;      // a preset the model trained on
    juce::Colour loner;     // a preset alone in its component: a hard snap
    juce::Colour edge;      // the mutual-kNN graph
    juce::Colour accent;    // what is sounding, and where it is going
    juce::Colour accent2;   // where you are pointing
    juce::Colour path;      // the drawn trajectory
    juce::Colour test;      // held out -- never seen in training
    juce::Colour warn;      // something went wrong (a dropout, a clip)

    /** The spectrogram's colormap, five stops.

        A real colormap rather than background-to-ink: the scientific ones
        exist because a single hue wastes most of its dynamic range on shades
        the eye cannot separate, and a spectrogram is exactly the case where
        that range is the whole point.
    */
    std::array<juce::Colour, 5> ramp;

    /** Ink at partial opacity -- for structure, which carries no meaning. */
    juce::Colour tint (float alpha) const { return ink.withAlpha (alpha); }
    /** Background at partial opacity, for scrims over the map. */
    juce::Colour shade (float alpha) const { return background.withAlpha (alpha); }

    /** An opaque colour `level` of the way along the ramp.

        Opaque matters: the spectrogram scrolls its strip sideways and paints
        one fresh column per frame, so a translucent fill would composite over
        whatever scrolled underneath and smear the history.
    */
    juce::Colour rampAt (float level) const
    {
        const float t = juce::jlimit (0.0f, 1.0f, level) * (float) (ramp.size() - 1);
        const auto low = (size_t) t;
        const auto high = juce::jmin (ramp.size() - 1, low + 1);
        return ramp[low].interpolatedWith (ramp[high], t - (float) low);
    }
};

inline constexpr int kPaletteCount = 5;

/** Index into kPalettes. Kept as its own type so the parameter, the menu and
    the drawing code cannot disagree about what 2 means. */
enum PaletteId { kMono = 0, kEmber = 1, kReef = 2, kDusk = 3, kPaper = 4 };

inline const Palette& palette (int id)
{
    // Built once, on first use: juce::Colour is not constexpr-constructible,
    // and a function-local static keeps the table beside the thing that needs
    // it rather than in a translation unit of its own.
    static const std::array<Palette, kPaletteCount> table = { {
        // Monochrome is the one theme that stays monochrome: every role is a grey.
        { "mono", "Monochrome",
          juce::Colour (0xff000000), juce::Colour (0xff0b0b0b), juce::Colour (0xff1f1f1f),
          juce::Colour (0xff141414), juce::Colour (0xfff2f2f2), juce::Colour (0xff9a9a9a),
          juce::Colour (0xff6e6e6e),
          juce::Colour (0xffd8d8d8), juce::Colour (0xff565656), juce::Colour (0xfff2f2f2),
          juce::Colour (0xffffffff), juce::Colour (0xffbdbdbd), juce::Colour (0xffe0e0e0),
          juce::Colour (0xff8f8f8f), juce::Colour (0xffffffff),
          { juce::Colour (0xff000000), juce::Colour (0xff343434), juce::Colour (0xff7c7c7c),
            juce::Colour (0xffc6c6c6), juce::Colour (0xffffffff) } },

        // Ember. Magma ramp, which spends its range on hues the eye separates
        // easily, with a cyan focus that cannot be confused with anything in it.
        { "ember", "Ember",
          juce::Colour (0xff0b0510), juce::Colour (0xff150a1e), juce::Colour (0xff34204a),
          juce::Colour (0xff1f1430), juce::Colour (0xfff4e9f8), juce::Colour (0xffb795c6),
          juce::Colour (0xff7d5f8e),
          juce::Colour (0xffffb37a), juce::Colour (0xff6f4a66), juce::Colour (0xffa763d8),
          juce::Colour (0xff35e0d3), juce::Colour (0xffffc65e), juce::Colour (0xffff6fae),
          juce::Colour (0xffc2f549), juce::Colour (0xffff4d4d),
          { juce::Colour (0xff04020c), juce::Colour (0xff57106e), juce::Colour (0xffbb3754),
            juce::Colour (0xfff98e08), juce::Colour (0xfffcfdbf) } },

        // Reef. Viridis, which stays legible converted to grey -- the one
        // colormap that survives being printed or screenshotted badly.
        { "reef", "Reef",
          juce::Colour (0xff04121a), juce::Colour (0xff072028), juce::Colour (0xff0f3f4a),
          juce::Colour (0xff082a33), juce::Colour (0xffe4f7f3), juce::Colour (0xff8fc7c1),
          juce::Colour (0xff55908b),
          juce::Colour (0xff7fe3c4), juce::Colour (0xff2b6b61), juce::Colour (0xff2fa8b8),
          juce::Colour (0xffff6b57), juce::Colour (0xffffd447), juce::Colour (0xffb78cff),
          juce::Colour (0xffd9f7ff), juce::Colour (0xffff3b30),
          { juce::Colour (0xff440154), juce::Colour (0xff3b528b), juce::Colour (0xff21918c),
            juce::Colour (0xff5ec962), juce::Colour (0xfffde725) } },

        // Dusk. Indigo night, blue through magenta, lime focus. The coolest
        // ground of the set, so the warm accents carry the whole foreground.
        { "dusk", "Dusk",
          juce::Colour (0xff06081c), juce::Colour (0xff0c1030), juce::Colour (0xff222a63),
          juce::Colour (0xff141a42), juce::Colour (0xffe9ecff), juce::Colour (0xff9ba4da),
          juce::Colour (0xff6570a2),
          juce::Colour (0xff8fa6ff), juce::Colour (0xff39417d), juce::Colour (0xff7b6ae0),
          juce::Colour (0xffb9ff3d), juce::Colour (0xffff8ad8), juce::Colour (0xffffd166),
          juce::Colour (0xff7ef5ff), juce::Colour (0xffff4d6d),
          { juce::Colour (0xff080a28), juce::Colour (0xff2e288c), juce::Colour (0xff8237b4),
            juce::Colour (0xffe05496), juce::Colour (0xffffd6ec) } },

        // Paper. Light and still coloured -- an inverted ramp, because on a
        // white ground the loud parts have to get darker, not brighter.
        { "paper", "Paper (light)",
          juce::Colour (0xfff4f1e8), juce::Colour (0xffe8e3d5), juce::Colour (0xffc4bdab),
          juce::Colour (0xffdbd5c5), juce::Colour (0xff1b1a16), juce::Colour (0xff55513f),
          juce::Colour (0xff857f6c),
          juce::Colour (0xff1f3a5f), juce::Colour (0xffa69e8b), juce::Colour (0xff5b7fa8),
          juce::Colour (0xffc1272d), juce::Colour (0xff16708b), juce::Colour (0xff7a4fa3),
          juce::Colour (0xff2e7d32), juce::Colour (0xffc1272d),
          { juce::Colour (0xfff4f1e8), juce::Colour (0xfff5d596), juce::Colour (0xffe89254),
            juce::Colour (0xffc1423a), juce::Colour (0xff4a1024) } },
    } };

    return table[(size_t) juce::jlimit (0, kPaletteCount - 1, id)];
}

}  // namespace atlasflow

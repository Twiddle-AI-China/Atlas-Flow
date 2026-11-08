#pragma once

#include <juce_core/juce_core.h>

namespace atlasflow {

/*  Both languages the desktop instrument speaks.

    The web demo carries its translations as data-en attributes with Chinese as
    the base text; this is the same pairing, in the same two languages, for the
    much smaller set of strings a plugin window actually shows.

    What stays untranslated is deliberate and matches the desktop build: the
    scope tags (SPECTROGRAM, WATERFALL, ...) are "the ASCII half of the label
    in either language", preset ids are identifiers, and dB/Hz/percent are
    units. Translating any of those would make the two builds disagree about
    what a screenshot says.
*/
enum Language { kEnglish = 0, kChinese = 1 };
inline constexpr int kLanguageCount = 2;

struct Phrase { const char* en; const char* zh; };

/** UTF-8 has to be announced. A bare const char* is read in the platform's
    default encoding, which on Windows turns every Chinese string into mojibake
    and on macOS turns a middot into "Â·". */
inline juce::String tr (const Phrase& phrase, int language)
{
    return juce::String::fromUTF8 (language == kChinese ? phrase.zh : phrase.en);
}

/** Which language to open in, when the session has not said.

    A plugin inherits its surroundings rather than choosing for the user: the
    desktop app can default to Chinese because it is its own window, but a
    plugin lands inside a host that is already in some language and should not
    be the one odd panel. Once chosen in settings the choice is stored with the
    session and this is not consulted again.
*/
inline int systemLanguage()
{
    return juce::SystemStats::getUserLanguage().startsWithIgnoreCase ("zh") ? kChinese : kEnglish;
}

namespace text {

// --- the strip -------------------------------------------------------------
inline constexpr Phrase draw       { "DRAW", "\xe7\x94\xbb\xe8\xbd\xa8\xe8\xbf\xb9" };            // 画轨迹
inline constexpr Phrase clear      { "CLEAR", "\xe6\xb8\x85\xe9\x99\xa4" };                        // 清除
inline constexpr Phrase settings   { "SETTINGS", "\xe8\xae\xbe\xe7\xbd\xae" };                     // 设置
inline constexpr Phrase voices     { "voices", "\xe5\xa3\xb0\xe9\x83\xa8" };                        // 声部
inline constexpr Phrase bank       { "bank", "\xe9\xa2\x84\xe8\xa7\xa3\xe7\xa0\x81\xe5\xba\x93" };  // 预解码库
inline constexpr Phrase path       { "path", "\xe8\xbd\xa8\xe8\xbf\xb9" };                          // 轨迹
inline constexpr Phrase planning   { "planning", "\xe8\xa7\x84\xe5\x88\x92\xe4\xb8\xad" };          // 规划中
inline constexpr Phrase folded     { "folded", "\xe5\x85\xab\xe5\xba\xa6\xe6\x8a\x98\xe5\x8f\xa0" };// 八度折叠

// --- the scope -------------------------------------------------------------
inline constexpr Phrase stream     { "stream", "\xe6\xb5\x81" };                                    // 流
inline constexpr Phrase peak       { "peak", "\xe5\xb3\xb0\xe5\x80\xbc" };                          // 峰值

// --- the info card ---------------------------------------------------------
inline constexpr Phrase trained    { "trained", "\xe5\xb7\xb2\xe8\xae\xad\xe7\xbb\x83" };           // 已训练
inline constexpr Phrase heldTest    { "held out (test)",
    "held-out\xef\xbc\x88\xe6\xb5\x8b\xe8\xaf\x95\xef\xbc\x89" };                                   // held-out（测试）
inline constexpr Phrase heldValid   { "held out (validation)",
    "held-out\xef\xbc\x88\xe9\xaa\x8c\xe8\xaf\x81\xef\xbc\x89" };                                   // held-out（验证）
inline constexpr Phrase alone       { "alone in its colony",
    "\xe8\x87\xaa\xe6\x88\x90\xe4\xb8\x80\xe4\xb8\xaa\xe8\x81\x9a\xe8\x90\xbd" };                   // 自成一个聚落
inline constexpr Phrase colonyOf    { "colony of ", "\xe8\x81\x9a\xe8\x90\xbd\xe5\xa4\xa7\xe5\xb0\x8f " }; // 聚落大小
inline constexpr Phrase nearest     { "nearest ", "\xe6\x9c\x80\xe8\xbf\x91 " };                    // 最近
inline constexpr Phrase drawable    { "% of that gap is drawable",
    "% \xe7\x9a\x84\xe8\xb7\x9d\xe7\xa6\xbb\xe7\x94\xbb\xe5\xbe\x97\xe5\x87\xba\xe6\x9d\xa5" };     // % 的距离画得出来

// --- settings --------------------------------------------------------------
inline constexpr Phrase language   { "language", "\xe8\xaf\xad\xe8\xa8\x80" };                      // 语言
inline constexpr Phrase themeName  { "theme", "\xe9\x85\x8d\xe8\x89\xb2" };                         // 配色
inline constexpr Phrase topPane    { "top", "\xe4\xb8\x8a\xe6\xa0\xbc" };                      // 上格
inline constexpr Phrase middlePane { "middle", "\xe4\xb8\xad\xe6\xa0\xbc" };                   // 中格
inline constexpr Phrase bottomPane { "bottom", "\xe4\xb8\x8b\xe6\xa0\xbc" };                   // 下格

inline constexpr Phrase aboutTheme
{
    "A theme changes the palette and nothing else: same hairlines, same layout, "
    "same information.",
    "\xe6\x8d\xa2\xe9\x85\x8d\xe8\x89\xb2\xe5\x8f\xaa\xe6\x8d\xa2\xe8\xb0\x83\xe8\x89\xb2\xe6\x9d\xbf"
    "\xef\xbc\x8c\xe5\xb8\x83\xe5\xb1\x80\xe5\x92\x8c\xe4\xbf\xa1\xe6\x81\xaf\xe9\x83\xbd\xe4\xb8\x8d"
    "\xe5\x8f\x98\xe3\x80\x82"
};
inline constexpr Phrase aboutScope
{
    "The frequency axis stops at 4 kHz -- measured, these pads put nearly all "
    "their energy below 1.3 kHz. The header states the stream's own ceiling "
    "beside it.",
    "\xe5\xae\x9e\xe6\xb5\x8b\xe8\xbf\x99\xe6\x89\xb9 pad \xe7\x9a\x84\xe8\x83\xbd\xe9\x87\x8f"
    "\xe5\x87\xa0\xe4\xb9\x8e\xe9\x83\xbd\xe5\x9c\xa8 1.3 kHz \xe4\xbb\xa5\xe4\xb8\x8b\xef\xbc\x8c"
    "\xe6\x89\x80\xe4\xbb\xa5\xe9\xa2\x91\xe7\x8e\x87\xe8\xbd\xb4\xe5\xb0\x81\xe9\xa1\xb6 4 kHz"
    "\xef\xbc\x9b\xe8\xa1\xa8\xe5\xa4\xb4\xe4\xbc\x9a\xe5\x90\x8c\xe6\x97\xb6\xe6\xa0\x87\xe5\x87\xba"
    "\xe5\xae\x9e\xe9\x99\x85\xe6\xb5\x81\xe7\x9a\x84\xe4\xb8\x8a\xe9\x99\x90\xe3\x80\x82"
};
inline constexpr Phrase aboutParameters
{
    "These are the host's parameters, not a second set: moving one here writes the same automation lane, into the same undo history.",
    "\xe8\xbf\x99\xe4\xba\x9b\xe5\xb0\xb1\xe6\x98\xaf\xe5\xae\xbf\xe4\xb8\xbb\xe9\x87\x8c\xe7\x9a\x84\xe9\x82\xa3\xe4\xba\x9b\xe5\x8f\x82\xe6\x95\xb0\xef\xbc\x8c\xe4\xb8\x8d\xe6\x98\xaf\xe5\x8f\xa6\xe4\xb8\x80\xe5\xa5\x97\xef\xbc\x9a\xe5\x9c\xa8\xe8\xbf\x99\xe9\x87\x8c\xe6\x8b\xa8\xe5\x8a\xa8\xe5\x86\x99\xe7\x9a\x84\xe6\x98\xaf\xe5\x90\x8c\xe4\xb8\x80\xe6\x9d\xa1\xe8\x87\xaa\xe5\x8a\xa8\xe5\x8c\x96\xe8\xbd\xa8\xe9\x81\x93\xef\xbc\x8c\xe8\xbf\x9b\xe5\x90\x8c\xe4\xb8\x80\xe4\xb8\xaa\xe6\x92\xa4\xe9\x94\x80\xe5\x8e\x86\xe5\x8f\xb2\xe3\x80\x82"
};


// --- the performance controls, which are also parameters ------------------
inline constexpr Phrase sectionLook { "APPEARANCE", "\xe5\xa4\x96\xe8\xa7\x82" };   // 外观
inline constexpr Phrase sectionPlay { "PERFORMANCE", "\xe6\xbc\x94\xe5\xa5\x8f" };   // 演奏
inline constexpr Phrase sectionScope{ "SCOPE PANES", "\xe9\xa2\x91\xe8\xb0\xb1\xe7\xaa\x97\xe6\xa0\xbc" };   // 频谱窗格
inline constexpr Phrase level       { "level", "\xe5\x8a\x9b\xe5\xba\xa6" };   // 力度
inline constexpr Phrase morph       { "morph", "\xe6\xb8\x90\xe5\x8f\x98" };   // 渐变
inline constexpr Phrase lap         { "lap", "\xe5\x91\xa8\xe6\x9c\x9f" };   // 周期
inline constexpr Phrase voiceCount  { "voices", "\xe5\xa3\xb0\xe9\x83\xa8" };   // 声部
inline constexpr Phrase temperature { "temperature", "\xe6\xb8\xa9\xe5\xba\xa6" };   // 温度
inline constexpr Phrase seed        { "seed", "\xe7\xa7\x8d\xe5\xad\x90" };   // 种子

}  // namespace text
}  // namespace atlasflow

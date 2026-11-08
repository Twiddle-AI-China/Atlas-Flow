# Atlas Flow — plugin

A VST3 / AU / Standalone build of the Atlas Flow timbre map, for Ableton Live,
FL Studio and anything else that hosts plugins. The host owns the MIDI; the map
owns the timbre.

> **The sound is currently a mock.** Every anchor on the map is a fixed mix of
> three basic waveforms — sine, saw, square — chosen deterministically from the
> preset index. There is no neural network here yet. What *is* real is the
> geometry: the coordinates, the scale and the mutual-kNN graph are the ones the
> trained model was built against, so the map you navigate is authentic from day
> one and only the timbre at each point is fake.
>
> The point of the mock is that the MIDI path, the voice allocation, the
> parameters and the UI can be finished and judged before the ONNX decoder
> lands. See the design proposal for what replaces it.

This directory is **separate from `electron/`**, which is the existing desktop
app and is not affected by anything here.

## Building

Needs CMake ≥ 3.22 and a C++17 toolchain. On macOS the Command Line Tools are
enough — a full Xcode install is not required, though without it CMake's Xcode
generator is unavailable, so use Makefiles or Ninja. On Windows, Visual Studio
2022 (the Desktop C++ workload) and a bash — Git Bash is what the fetch script
is written against.

```bash
git submodule update --init --depth 1 plugin/third_party/JUCE   # once: JUCE 9.0.3
bash plugin/tools/fetch_onnxruntime.sh                          # once: ONNX Runtime 1.23.2
cd plugin
cmake -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

On Windows the generator is multi-config, so the configuration is named at
build time rather than at configure time, and the architecture is passed to the
generator:

```bash
cmake -B build -A x64
cmake --build build --config Release
```

ONNX Runtime is pinned, not "latest": the exported graphs were verified against
1.23.2 in Python, and a runtime mismatch is the kind of thing that changes audio
without changing any source file. Neither it nor JUCE is in the tree.

Products land in `build/AtlasFlow_artefacts/Release/` and, because
`COPY_PLUGIN_AFTER_BUILD` is on, the VST3 and AU are also installed into
`~/Library/Audio/Plug-Ins/`. Build a single format with
`--target AtlasFlow_VST3`, `AtlasFlow_AU` or `AtlasFlow_Standalone`.

### What differs on Windows

AU is a macOS format and is dropped from the build there, so Windows gets VST3
and Standalone. The rest is one difference with several consequences: Windows
has no rpath. A DLL is resolved from the directory of the module that imported
it, never from a path recorded inside the binary, so `onnxruntime.dll` is
*copied* next to each product instead of being referenced the way the macOS
build references its dylib — into `Contents/x86_64-win/` for the VST3, beside
the `.exe` for the standalone, and beside both console tools. A VST3 is still a
bundle on Windows, so the exported graphs sit in `Contents/Resources/models`
exactly as they do on macOS; only the standalone differs, keeping them in a
`models/` folder beside the executable.

There is no code signing step, which removes the entire class of failure the
macOS build has to work around below.

**Not yet verified on real hardware.** Every line of this port was written and
checked on macOS; Windows is covered by the `windows-latest` CI job, which
builds both products, asserts the DLL landed beside each one, and runs the
render check against the mock engine. The model engine has never been run on
Windows — CI does not export the graphs, for the storage reason the workflow
explains — so ONNX on Windows remains untested.

## Testing

```bash
cmake --build build --target AtlasFlowRenderTest AtlasFlowHostTest -j 8
./build/AtlasFlowRenderTest_artefacts/Release/AtlasFlowRenderTest
```

A plugin that builds proves nothing about whether it makes the right sound, so
this renders notes offline with no host and asserts on the things that would
otherwise only be caught by ear: pitch, octave folding, the voice ceiling, and
whether moving across the map changes the timbre without changing the pitch.

The last section checks the surface, because a claim drawn wrongly is worse than
one not drawn at all — it still looks authoritative. That the rings land on the
three presets `atlas_flow_manifest` holds out; that the terrain has relief and
that a singleton preset sits lower on it than a colony does; that the field
agrees with the desktop instrument's own version of it; that every palette's
ramp is monotone in brightness and legible; that the analyser finds a tone where
it is and scales it as an `AnalyserNode` would; and that all five views actually
draw. The field agreement is measured at **r = 0.952** over 1024 points, which
is what the 0.95 gate is set just under — the two differ only in the metric used
to rank neighbours, and a regenerated atlas that pushed them apart is exactly
what the gate is there to catch.

All of that builds the processor directly, which never touches the wrapper a
user actually installs. `AtlasFlowHostTest` does the other half: it links no
Atlas Flow code at all and loads the built bundle through
`juce::AudioPluginFormat`, driving it through the same generic
`AudioPluginInstance` interface Logic or Live has.

```bash
./build/AtlasFlowHostTest_artefacts/Release/AtlasFlowHostTest \
    --vst3 "build/AtlasFlow_artefacts/Release/VST3/Atlas Flow.vst3"
```

Twelve checks per format, run against VST3 and — on macOS, from the system
registry, which is the only way the OS will load one — AU: that it scans, that
the host reads the right name, that it takes MIDI, that it reports no latency,
that silence stays silent and a note sounds, that the parameter list is sane in
both size and text, that the state blob round-trips through the wrapper's own
path, and that the editor can be created and painted out of the loaded binary.

It earned its place immediately. The VST3 was publishing **2089 parameters** —
2080 of them JUCE's MIDI CC emulation, one per controller per channel, which is
on by default and fills a host's parameter list and automation menu with noise
for a synth that ignores CC. Turned off, the count is 9. Nothing else in this
directory could have seen that, because nothing else asks the wrapper what it
tells a host.

None of that would notice a window with everything stacked in one corner, so
there is also a mode that paints the editor offline, once per palette:

```bash
./build/AtlasFlowRenderTest_artefacts/Release/AtlasFlowRenderTest --snapshot shots/
```

It needs no window server and takes no screenshot, so it runs anywhere and the
PNGs can be attached to a review. It drives the editor at its own 30 Hz in real
time and paints every frame, because a scope scrolls *inside* `paint()` and a
loop that renders five seconds of audio in a fiftieth of a second photographs
three empty panes. It is also the Windows CI's only coverage of the editor.

## The planner

`atlasflow-plan` renders one plan offline — a coordinate on the map becomes five
seconds of audio — and is the thing that makes the hand-ported half trustworthy.
The two networks are identical by construction on both sides, so any difference
from the Python runtime comes from the parts written by hand: the atlas
projection, the kNN routing, the smoothstep navigation, the lifecycle signal and
the Heun solver loop.

```bash
cmake --build build --target atlasflow-plan -j 8
PYTHONPATH=src python plugin/tools/compare_with_python.py
```

All seven cases match, across both plan modes, five pitches, restarts and
routed morphs:

```
x=+0.0 y=+0.0 n=50       graph_plan             traj -113.0dB  audio -105.3dB  PASS
x=+0.2 y=-0.4 n=50       cross_component_plan   traj -118.4dB  audio -107.2dB  PASS
x=+0.9 y=+0.9 n=71       graph_plan             traj -122.2dB  audio -115.9dB  PASS
x=+0.2 y=-0.4 n=50 <-19  graph_plan             traj -117.4dB  audio -107.1dB  PASS
```

The playback layer is ported too (`src/Voice.h`): the sustain loop, the morph
crossfade and the release tail, so a note can be held past the five-second
render. Two playback cases are checked the same way, holding well past 220500
samples so the loop has to engage — `-108.2 dB` and `-109.3 dB` RMS against the
Python session, block for block.

One deliberate limit: **temperature is 0 only**. The solver's initial residual is
noise scaled by temperature, so at zero it vanishes and both sides are
deterministic; supporting temperature > 0 means reproducing torch's Philox
stream in C++, which is separate work.

## The real engine in the plugin

`PluginProcessor` runs the model when the exported graphs are present and falls
back to the mock when they are not, so the plugin always loads and always makes
a navigable sound. The status line says which: `[MODEL]` or `[MOCK]`.

Graphs are looked for in `$ATLASFLOW_MODELS` (which wins outright — "look here"
that silently looks elsewhere wastes an afternoon; set it to `none` to force the
mock), then the path baked in at build time, then
`~/Library/Application Support/AtlasFlow/models`.

**Threading.** A plan costs ~700 ms and the callback has ~11 ms, so planning
happens on a worker thread and the callback only reads PCM that already exists.
The handoff is not a `shared_ptr` swap: releasing the last reference would free
about a megabyte on the audio thread. The service owns every plan it has made in
a four-deep ring and publishes a borrowed raw pointer; the audio thread never
touches a reference count, never allocates and never frees.

**Latency is real, and it is aligned.** The processor writes into an output
ring addressed by *absolute sample position*, and voices add into it rather than
overwrite, so a chord sums correctly even when its notes' plans finish at
different moments. When a plan lands late the note's audio is written back to
where the note-on was — that position has not been read yet — rather than
appended at "now". Without this the plan time stacks on top of the latency the
host already compensated and every note in a sequenced part lands late. The
headless check asserts it: `onset 85995, latency 85995, drift 0`.

**Voices are a trade against latency, and the plugin measures its own speed to
size it.** The flow predictor never sees MIDI, so the trajectory is cached
against the position that produced it and a chord costs **one flow and N
decodes** rather than N of each. On an M2, warm: flow ~105 ms, decode ~395 ms.

The latency is not a constant. At load the service runs each graph once and
times the decode, and `prepareToPlay` sizes the report from that measurement
plus a quarter margin. This matters because **the same binary runs about 1.8×
slower under Rosetta** — 655 ms against 1187 ms for one plan here — and a
latency sized from the native figure is a deadline the translated build cannot
meet. A host with Rosetta enabled is not hypothetical; it is what this plugin
was first installed into. Measured, four voices:

| | reported latency |
|---|---|
| native arm64 | 92252 samples (2.09 s) |
| under Rosetta | 170878 samples (3.87 s) |

Both land with `drift 0`.

Block-streaming the decoder is what would collapse this: 41 ms per 16384
samples puts four voices inside real time, and the exported graph already takes
the `window_start` input that needs. It is the proper fix and is not done yet.

**The session is warmed at load.** ONNX Runtime allocates arenas and picks
kernels on the first call — 837 ms cold against 393 ms warm for a decode — so
both graphs are run once on throwaway input before any note arrives. That pays
the cost at load instead of on the user's first note, and is what lets the
reported latency be sized from the warm figure rather than the worst one.

**Offline rendering blocks.** A bounce runs faster than real time, so a
background plan would never land and the render would come out silent. When
`isNonRealtime()` the processor waits for the planner, which is safe offline and
forbidden in the callback.

`auval -v aumu Atfl Twdl` passes, including its render, parameter-scheduling and
MIDI tests — and it still passes with `third_party/onnxruntime` renamed away,
which is the only way to prove the bundle is actually self-contained rather than
quietly loading the copy in the build tree.

**The bundles are self-contained: 161 MB each.** ONNX Runtime goes into
`Contents/Frameworks` and the exported graphs into `Contents/Resources/models`,
and the binary carries `@loader_path/../Frameworks`. Nothing reaches outside the
bundle. A plugin that does — at an rpath into someone's build tree, or a
`models/` folder in a checkout — works exactly once, on the machine it was built
on, and fails everywhere else with no useful message.

| | |
|---|---|
| executable (universal) | 10 MB |
| ONNX Runtime (arm64 33 MB + x86_64 30 MB) | 63 MB |
| exported graphs | 88 MB |

Against the Electron desktop build's 933 MB, about a sixth.

Two things that bite here. Editing a Mach-O invalidates its signature, and
macOS answers an invalid signature with `SIGKILL` and no message at all, so the
bundle is re-signed ad hoc after the embed. And JUCE's own install step is also
`POST_BUILD` and runs *before* ours, so the embed is applied to the installed
copy in `~/Library/Audio/Plug-Ins` too — otherwise that one, the one a host
actually loads, would be the broken one.

Verified by hiding both `third_party/onnxruntime` and `models/` and running
`auval` again. That is the only way to tell a self-contained bundle from one
quietly loading the build tree.

## What it does today

| | |
|---|---|
| **MIDI in** | Note on/off from the host, sample-accurate within the block. All-notes-off honoured. |
| **Voices** | Four, with oldest-note stealing. The model this will host is monophonic — a chord is four independent renders summed — and four voices measured about 1.65× real time on CPU, so the ceiling is real rather than arbitrary. |
| **Note range** | The trained range is MIDI 36–71. Notes outside it are **folded by octaves**, not clamped. Clamping is what the Python runtime does today, and in a DAW it silently retunes the note, which is wrong. |
| **Velocity** | Output gain only — as in the trained model, which does not use velocity for timbre. |
| **Map** | `atlasX` / `atlasY` are ordinary automatable parameters. Dragging on the map writes them through the host, so a gesture here and an automation lane are the same thing. |
| **Drawn path** | `DRAW`, then drag: the stroke becomes the timbre automation, walked there and back, `Lap` seconds one way. The speed profile is curvature-based — tight turns taken slowly, normalised against the median so one scribbled hairpin colours the walk instead of halting it, eased at both ends, and scaled by its harmonic mean so the lap time you asked for is the one you get. `CLEAR` hands control back to the parameters. |
| **Parameters** | `Morph` 0.5–5 s, `Temperature`, `Seed`, `Lap`, `Voices`, `Level`. Each states its own units and precision, so a host's generic editor and automation lane read `2.0 s` rather than `2.000000`. Above zero, temperature stops being bit-identical to the Python runtime by design: matching torch would mean reimplementing its Philox stream, and a different draw from the same distribution is an equally valid sample. The seed still makes it reproducible here. |
| **Edges** | The k=3 nearest-neighbour graph the desktop build draws, 121 pairs, split by what a move across one actually does: **48 roads** (a direct mutual-kNN hop, in the edge hue), **19 reachable** (no direct road, but the graph gets there — neutral, dotted), and **54 jumps** (a different colony, so no path exists and the voice restarts — in the loner hue, the colour a stranded preset already wears). Brightness is true 8-D closeness; dashed marks the worst quarter of projection loss. |
| **Settings** | Everything that is not the map, behind one gear: appearance, the performance parameters, and which view sits in each scope pane. |
| **Terrain** | The background is how much the runtime will overrule you, relief-shaded and contoured. Bright is what-you-see-is-what-you-hear; the dark basins are where a coordinate gets pulled into legal territory first. Measured through the planner's own `projectDetails`, not a copy of it. |
| **Scopes** | Three panes, each showing any of five views — spectrogram, waterfall, loudness, spectrum, wave — fed by one analyser pass per frame off a lock-free tap on the plugin's actual output. |
| **Appearance** | Five palettes and two languages. Stored on the state tree rather than as parameters, so changing them does not write an automation lane. |
| **Output** | Mono, copied to both channels, as the trained model is also mono. |

Latency is reported as **zero samples**, and measured as zero: a note is served
from the pre-decoded pitch bank, so there is nothing for the host to compensate.
The flow predictor never sees MIDI, which is what makes that bank possible --
one trajectory covers all 36 pitches. What it costs is elsewhere: a map move
retires the whole bank, so the first note after roaming waits for a fresh plan
(~490 ms here) and a full bank takes ~14 s to refill at 388 ms per pitch. On a
held note that wait hides under the morph crossfade; on a new note it does not.

## The surface

The desktop instrument and this editor draw the same map, and the point of that
is that a screenshot of one explains the other. Everything visual in
`atlas-flow-web-demo/` is here: the terrain and its iso-fidelity contours, the
sparse lattice, the labelled PC1/PC2 rails, the graph with its folded edges, the
held-out rings, singleton presets drawn smaller, the double ring on what is
sounding, the dashed reticle on the voice with a leash back to the pointer, the
hover label and info card, the 8-D inset, the three scope panes and their five
views, the five palettes, and the gear that holds all of it.

Behind the gear: language and palette; `Level`, `Morph`, `Lap`, `Voices`,
`Temperature` and `Seed`; and the three pane assignments. The parameters are
attached to the APVTS rather than read from it, so moving one here and
automating it in the host are the same act in the same undo history. Language,
palette and pane assignment are *not* parameters — they are properties on the
state tree, because automating a change of mind about colours is not something
anyone wants in a lane. All of it is saved with the session either way.

Four things differ on purpose:

- **Hover names a preset; it does not play one.** On the web, moving the pointer
  *is* the audition, which is right for a page you opened to explore. In a host
  it would write two automation parameters every time the mouse crossed the
  window, so here the pointer has to be pressed to mean anything.
- **The loudness pane has two traces, not three.** The third is the desktop
  build's multiband compressor's gain reduction. There is no compressor here:
  levelling belongs in the host's chain, and a plugin that quietly levels its
  own output is lying about the model's loudness — which is one of the gates the
  model does not pass.
- **The terrain is aligned to the atlas.** The web stretches the field across
  its whole canvas while its presets sit inside a 64 px pad, so there the ground
  is offset from the map it describes by the width of that pad. Here both are
  drawn on the same rectangle.
- **A drawn edge says what a move across it costs.** The desktop build draws a
  k=3 nearest-neighbour graph in which the atlas looks like one connected
  object. The planner routes on the atlas's *mutual* kNN instead — 48 edges
  over **14 connected components** — and asks one question of a move: same
  component or not. That splits the 121 drawn pairs three ways: 48 are a direct
  road, 19 are near with no direct road but a route through the graph, and
  **54 are near and in a different colony**, so a move across one restarts the
  voice every single time. The demo's caption gestures at this — "that is
  usually where roaming jumps" — but every line in its picture looks alike, so
  it cannot say which. Here hue carries it.

Not ported, because they are a server or a host rather than a frontend: the
audition-mode/wire/buffer/output-rate settings (there is no stream — the model
is in-process), the chord-progression player and loop (the host sequences MIDI),
the multiband compressor and its meters, and take recording and offline render.

The analyser is a deliberate reimplementation of a Web Audio `AnalyserNode`
rather than an improvement on one: every constant in the views — the 1.2
exponent on the spectrogram, the 0.15 floor on the waterfall, where the loudness
traces sit — is tuned to what that node returns, so a "better" FFT with a
different window or dB range would move all of them. Blackman window, fftSize
2048, smoothing 0.45, magnitudes over fftSize, mapped across −100…−30 dB, and
`render_test` asserts it lands within 0.04 of the figure that specification
predicts.

## Layout

| | |
|---|---|
| `src/Atlas.h` | Map geometry: screen placement, and the inverse-distance blend over nearby presets |
| `src/AtlasData.h` | **Generated** — the real 50-preset atlas. Do not edit; see below |
| `src/MockVoice.h` | The placeholder instrument: three waveforms, polyBLEP band-limited, and the octave folding |
| `src/PluginProcessor.*` | MIDI handling, voice allocation, parameters, state |
| `src/PluginEditor.*` | The map, the scope pane, the settings panel and the status line |
| `src/Palette.h` | The five palettes, ported from the demo's stylesheet |
| `src/Terrain.h` | The override field: height, relief shading, marching-squares contours |
| `src/Analyser.h` | The audio tap and a Web Audio `AnalyserNode`, reproduced |
| `src/Scopes.h` | The five scope views |
| `src/Strings.h` | English and Chinese, for the strings a plugin window shows |
| `tools/export_atlas.py` | Regenerates `AtlasData.h` from the trained atlas |
| `tools/render_test.cpp` | The headless check, and `--snapshot` |
| `tools/host_test.cpp` | Loads the built bundle the way a DAW does |

`AtlasData.h` is committed so the plugin builds without the model download. To
regenerate it after `scripts/atlas_flow/fetch_local_model.sh`:

```bash
python plugin/tools/export_atlas.py
```

## Replacing the mock

The graphs the plugin will load are produced by
[`scripts/atlas_flow/export_onnx.py`](../scripts/atlas_flow/export_onnx.py):

```bash
PYTHONPATH=src python scripts/atlas_flow/export_onnx.py     # -> plugin/models/
```

That writes `flow.onnx` and `decoder.onnx` (about 92 MB together, gitignored —
only `manifest.json` is tracked) and refuses to produce an export that has
drifted from torch. Measured against the trained weights: flow relative error
`2.07e-06`, decoder `-113.4 dB` RMS and `-88.1 dB` peak. CI runs the same gate
against a randomly-initialised model on every change to the model or the
exporter.

The Heun solver loop is deliberately **not** in the flow graph — four steps of
two evaluations, pure control flow with no learned parameters — so the host
language owns the loop, which is what this plugin needs anyway.

`MockVoice` is deliberately the only thing that knows how a preset sounds. The
contract the decoder has to meet is the same one it meets now — *given a note
and a blend over presets, fill a buffer* — so swapping it should not require
touching the processor or the editor. What will change is the threading: the
decoder cannot run in the audio callback, so the plan/render split from the
Python runtime comes with it, along with the reported latency.

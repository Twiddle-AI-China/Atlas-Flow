# The paper

**Atlas Flow: Explicit Timbre-Trajectory Modeling for a Navigable Latent-Space
Synthesizer**

Yuxuan Shi¹\*, Yifeng Huang²\*, Jiangnan Zhang³\*, Jinyue Guo⁴, Jiayi Hu⁵†

¹ University of Pennsylvania · ² Fudan University · ³ Shanghai Conservatory of
Music · ⁴ RITMO, University of Oslo · ⁵ TwiddleAI
<br>\* Equal contribution. † Corresponding author.

📄 **[atlas-flow-ismir2026-lbd.pdf](atlas-flow-ismir2026-lbd.pdf)** — 3 pages

> **On citing this.** This is an ISMIR **Late-Breaking Demo** extended abstract.
> Per the ISMIR LBD guidelines, extended abstracts are not part of the official
> conference proceedings and should not be cited as conference papers. The
> source here is built with the `submission` option.

## What is in it

The abstract, in one line: each coordinate on the map returns a *trajectory*
rather than a single latent vector, so timbre keeps moving through a held note,
and a path drawn across the map becomes timbre automation for the take.

The paper is the authoritative description of the **training procedure**, which
this repository implements but does not otherwise explain in prose:

- **Phase 1** (60k updates) — encoder and decoder, with the cross branch that
  forces pitch out of the timbre trajectory: A's timbre is asked to render B's
  note, so any pitch stored in `z` emerges at the wrong pitch where the
  reconstruction loss punishes it. Reinforced by a tie loss and a
  gradient-reversal pitch adversary.
- **Phase 2** (100k updates) — the flow predictor, trained with conditional flow
  matching on the residual against each preset's anchor, over a mix of full
  plans, continuations, and preset switches.
- **Phase 3** (30k updates) — joint.

Section 2.2 describes inference: how a pointer position becomes eight atlas
dimensions, how routing over the mutual 4-NN graph produces the anchor path, and
the four-step Heun solver in autoregressive 64-frame blocks.

## Rebuilding the PDF

Needs a TeX distribution with `pdflatex` and `bibtex` (TeX Live or MacTeX). All
style files the paper needs — `ismir.sty`, `cite.sty`, `IEEEtran.bst`, the
`cc_by` marks — are vendored here, so no package installation is required.

```bash
cd docs/paper
pdflatex atlas-flow-ismir2026-lbd.tex
bibtex   atlas-flow-ismir2026-lbd
pdflatex atlas-flow-ismir2026-lbd.tex
pdflatex atlas-flow-ismir2026-lbd.tex      # twice more, to settle references
```

A clean build is 3 pages with no undefined references or citations. The
intermediate `.aux`, `.bbl`, `.blg` and `.log` files are ignored by git.

## Layout

| | |
|---|---|
| `atlas-flow-ismir2026-lbd.tex` | The paper |
| `atlas-flow-ismir2026-lbd.pdf` | Built output, committed so the link in the README resolves without a TeX install |
| `ISMIRtemplate.bib` | Bibliography — 13 entries, all cited |
| `figures/` | Full-resolution figures |
| `ismir.sty`, `cite.sty`, `IEEEtran.bst`, `cc_by.*` | Vendored ISMIR 2026 template files, unmodified |

The only change to the template source is a `\graphicspath{{figures/}}` after
`\usepackage{graphicx}`: the ISMIR zip keeps figures flat beside the `.tex`,
and this repository keeps them in `figures/`.

## Figures

| | |
|---|---|
| `figures/atlas-flow-training.pdf` | The two training phases — encoder/decoder, and the flow predictor |
| `figures/atlas-flow-UI-paper-light.png` | The application UI, `paper` theme (Figure 2 in the paper) |

Both screenshots in the root README — this one and the monochrome variant —
are reproducible with `scripts/atlas_flow/capture_ui.js`; see
[the docs index](../README.md#static-resources).

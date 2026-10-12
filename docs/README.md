# Documentation

Everything that is not the front page. Start at the
[README](../README.md) if you just want to run the instrument.

## The paper

| | |
|---|---|
| [`paper/atlas-flow-ismir2026-lbd.pdf`](paper/atlas-flow-ismir2026-lbd.pdf) | *Atlas Flow: Explicit Timbre-Trajectory Modeling for a Navigable Latent-Space Synthesizer* — ISMIR 2026 Late-Breaking Demo |
| [`paper/`](paper/) | LaTeX source, bibliography, figures, and how to rebuild the PDF |

## Guides and specifications

| | |
|---|---|
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | Implementation notes for the training framework |
| [`TRAINING_VALIDATION_AND_INFRA_PLAN.md`](TRAINING_VALIDATION_AND_INFRA_PLAN.md) | Training validation and infrastructure plan |
| [`Atlas-Flow-v5-完整交接文档-2026-08-26.md`](Atlas-Flow-v5-完整交接文档-2026-08-26.md) | Full Atlas Flow v5 handoff: architecture, data, training, evaluation, operations, limitations, recovery |
| [`research-history.md`](research-history.md) | Earlier and parallel lines of work, kept for provenance |
| [`spec/`](spec/) | Design specifications |

## Static resources

| | |
|---|---|
| [`assets/`](assets/) | Images used by the README and the docs. Display-resolution copies (1600 px wide); the full-resolution originals live in [`paper/figures/`](paper/figures/) |
| [`evidence/`](evidence/) | Machine-readable evaluation evidence cited by the v5 handoff |

### Regenerating the README screenshots

Both are captures of the real application, not mockups, and
`scripts/atlas_flow/capture_ui.js` reproduces either one. Start a local server,
then drive it:

```bash
PYTHONPATH=src python -m midibrave.atlas_flow_local --port 18899 &

PORT=18899 THEME=paper OUT=/tmp/ui-paper.png \
  ./electron/node_modules/.bin/electron scripts/atlas_flow/capture_ui.js
PORT=18899 THEME=mono  OUT=/tmp/ui-mono.png \
  ./electron/node_modules/.bin/electron scripts/atlas_flow/capture_ui.js

sips -Z 1600 -s format png /tmp/ui-paper.png --out assets/atlas-flow-ui-paper-light.png
sips -Z 1600 -s format png /tmp/ui-mono.png  --out assets/atlas-flow-ui-monochrome.png
```

The script mutes audio three separate ways and the comment at the top explains
why. Do not weaken that to make a capture work.

### Adding a resource

Put images and other static files in `assets/`, and anything a document needs at
full resolution in `paper/figures/`. `.gitignore` keeps `docs/assets/**` and
`docs/evidence/**` explicitly un-ignored, so a new image under them is tracked
without further ceremony — but note the repository ignores `*.wav`, `*.flac`,
`*.mp3` and `*.ogg` everywhere, so audio examples need a deliberate decision
rather than a `git add`.

Keep display copies small. The README's two screenshots are 1600 px wide and
under 1.2 MB each; the 2938 px originals they came from are three times that,
and a README is read far more often than it is rebuilt.

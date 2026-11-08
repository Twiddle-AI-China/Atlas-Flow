"""Check the C++ planner against the Python runtime, sample for sample.

The two networks are identical by construction -- both sides run the same ONNX
graphs -- so any difference comes from the parts that were hand-ported: the
atlas projection, the kNN routing, the smoothstep navigation, the lifecycle
signal and the Heun solver loop. That is exactly the surface this checks.

Run after building `atlasflow-plan`:

    PYTHONPATH=src python plugin/tools/compare_with_python.py

Add --quick for a single case. Temperature is always 0: the solver's initial
residual is noise * temperature, so at zero it vanishes and both sides are
deterministic without needing to reproduce torch's RNG stream in C++.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Where the hand-ported code may differ from the reference before it is a bug.
# Both sides run the same graphs in the same runtime, so what is left is float
# ordering; the measured drift is around -107 dB RMS.
MAX_RMS_DB = -90.0
MAX_PEAK_DB = -70.0

# (x, y, note, hold blocks, tail blocks) -- drives the playback layer: the
# sustain loop, the release crossfade and the learned tail. Holding past
# 220500 samples is the point; that is where the loop has to engage.
PLAYBACK_CASES = [
    (0.2, -0.4, 50, 40, 20),
    (-0.6, 0.3, 60, 60, 24),
]

CASES = [
    # (x, y, note, source preset or None)
    (0.0, 0.0, 50, None),
    (0.2, -0.4, 50, None),
    (-0.8, 0.6, 36, None),
    (0.9, 0.9, 71, None),
    (0.2, -0.4, 50, 19),     # morph with a routed path
    (-0.5, 0.1, 60, 45),     # morph the other way
    (0.7, -0.7, 48, 0),
]


def db(value: float, peak: float) -> float:
    return 20.0 * np.log10(max(float(value), 1e-20) / max(peak, 1e-20))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "plugin" / "build" / "atlasflow-plan")
    parser.add_argument("--models", type=Path, default=ROOT / "plugin" / "models")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"not built: {args.binary}\n"
                         "  cmake --build plugin/build --target atlasflow-plan")

    warnings.filterwarnings("ignore")
    from midibrave.atlas_flow_local import build_parser, resolve
    from midibrave.atlas_flow_runtime import (
        AtlasFlowLiveEngine, AtlasFlowLiveSession, PlanRequest,
    )

    resolved = dict(zip(*[iter(resolve(build_parser().parse_args(["--print-argv"])))] * 2))
    engine = AtlasFlowLiveEngine.load(resolved["--config"], resolved["--checkpoint"], device="cpu")
    session = AtlasFlowLiveSession(engine)

    cases = CASES[:1] if args.quick else CASES
    playback = [] if args.quick else PLAYBACK_CASES
    failures = 0
    print(f"{'case':28s} {'mode':22s} {'traj rms':>10s} {'audio rms':>10s} {'peak':>9s}")

    with tempfile.TemporaryDirectory() as tmp:
        for x, y, note, source in cases:
            out = Path(tmp) / "cpp.f32"
            command = [str(args.binary), "--models", str(args.models),
                       "--x", str(x), "--y", str(y), "--note", str(note),
                       "--out", str(out)]
            if source is not None:
                command += ["--from-preset", str(source)]
            run = subprocess.run(command, capture_output=True, text=True)
            if run.returncode != 0:
                print(f"  C++ failed: {run.stderr.strip()[:200]}")
                failures += 1
                continue

            pca = [x, y, 0, 0, 0, 0, 0, 0]
            control = session._make_control(
                1, pca_normalized=pca, x=None, y=None, note=note,
                velocity=1.0, temperature=0.0, morph_seconds=2.0)
            restart = source is None
            index = engine.center_index if restart else source
            request = PlanRequest(
                1, control, 7,
                engine.atlas.coordinates[index].copy(),
                engine.atlas.anchors[index].copy(),
                engine.component_membership[index], restart)
            plan = session._compute_plan(request)

            reference = plan.waveform[:, 0].astype(np.float64)
            got = np.fromfile(out, dtype="<f4").astype(np.float64)
            reference_traj = plan.trajectory.numpy()
            got_traj = np.fromfile(str(out) + ".traj", dtype="<f4").reshape(128, 108).T

            if reference.shape != got.shape:
                print(f"  length mismatch {reference.shape} vs {got.shape}")
                failures += 1
                continue

            peak = float(np.abs(reference).max())
            audio_rms = db(np.sqrt(((reference - got) ** 2).mean()), peak)
            audio_peak = db(np.abs(reference - got).max(), peak)
            traj_peak_value = float(np.abs(reference_traj).max())
            traj_rms = db(np.sqrt(((reference_traj - got_traj) ** 2).mean()), traj_peak_value)

            ok = audio_rms <= MAX_RMS_DB and audio_peak <= MAX_PEAK_DB
            failures += 0 if ok else 1
            label = f"x={x:+.1f} y={y:+.1f} n={note}" + (f" <-{source}" if source is not None else "")
            print(f"{label:28s} {plan.mode:22s} {traj_rms:9.1f}dB {audio_rms:9.1f}dB "
                  f"{audio_peak:8.1f}dB  {'PASS' if ok else 'FAIL'}")

    # ---- playback layer ----------------------------------------------
    print(f"\n{'playback case':28s} {'blocks':>8s} {'rms':>10s} {'peak':>9s}")
    with tempfile.TemporaryDirectory() as tmp:
        for x, y, note, hold, tail in playback:
            out = Path(tmp) / "held.f32"
            run = subprocess.run(
                [str(args.binary), "--models", str(args.models),
                 "--x", str(x), "--y", str(y), "--note", str(note),
                 "--hold-blocks", str(hold), "--tail-blocks", str(tail),
                 "--out", str(out)],
                capture_output=True, text=True)
            if run.returncode != 0:
                print(f"  C++ failed: {run.stderr.strip()[:200]}")
                failures += 1
                continue

            voice = engine.new_session()
            voice.start(note=note, velocity=1.0, seed=7, temperature=0.0,
                        pca_normalized=[x, y, 0, 0, 0, 0, 0, 0])
            blocks = [voice.render_block()[0] for _ in range(hold)]
            voice.note_off()
            blocks += [voice.render_block()[0] for _ in range(tail)]
            reference = np.concatenate(blocks, axis=0)[:, 0].astype(np.float64)
            got = np.fromfile(out, dtype="<f4").astype(np.float64)
            count = min(len(reference), len(got))
            reference, got = reference[:count], got[:count]

            peak = float(np.abs(reference).max())
            rms = db(np.sqrt(((reference - got) ** 2).mean()), peak)
            worst = db(np.abs(reference - got).max(), peak)
            ok = rms <= MAX_RMS_DB and worst <= MAX_PEAK_DB
            failures += 0 if ok else 1
            label = f"x={x:+.1f} y={y:+.1f} n={note}"
            print(f"{label:28s} {hold}+{tail:<4d} {rms:9.1f}dB {worst:8.1f}dB  "
                  f"{'PASS' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"{failures} case(s) failed", file=sys.stderr)
        return 1
    total = len(cases) + (0 if args.quick else len(PLAYBACK_CASES))
    print(f"all {total} cases match the Python runtime "
          f"({len(cases)} plan, {total - len(cases)} playback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

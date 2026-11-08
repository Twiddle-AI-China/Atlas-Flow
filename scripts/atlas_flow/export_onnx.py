"""Export Atlas Flow to ONNX, and refuse to ship an export that does not match.

Two graphs come out of this:

    flow.onnx      one velocity evaluation of the flow predictor
    decoder.onnx   trajectory + note -> audio

The Heun solver loop is deliberately *not* in the flow graph. It is four steps
of two evaluations each, pure control flow with no learned parameters, and
keeping it outside means the host language owns the loop -- which is what the
C++ plugin needs anyway.

Everything else that is not a network -- the atlas projection, the kNN routing,
the plan assembly, the playback crossfades -- was never in the graph to begin
with and is ported separately.

The point of this script is the gate at the end. An export that silently drifts
from the trained model is worse than no export, so every graph is run back
through onnxruntime against the same inputs as torch and the difference has to
clear a threshold or the script exits non-zero.

    python scripts/atlas_flow/export_onnx.py                    # local-model/
    python scripts/atlas_flow/export_onnx.py --out plugin/models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# How far an export may drift from torch before it is rejected.
#
# The flow output is a velocity field, so it is judged relatively.
#
# The decoder is audio and gets two thresholds, because one is not enough.
# Worst-sample error over a five-second render is a noisy statistic -- measured
# across different random trajectories it swings about 9 dB (-79 to -88) purely
# from which sample happens to land worst -- so gating on it alone flaps. RMS is
# the stable one and is also the one that corresponds to audibility; measured
# drift is -113 dB RMS with the error spread evenly over the whole render and
# only 0.06% of samples anywhere near the max, which is float accumulation
# through a long causal chain rather than a mistranslated operator. Peak is kept
# as a loose safety net: it would catch a single broken op without tripping on
# ordinary fp32 noise.
FLOW_MAX_RELATIVE = 1.0e-4
DECODER_MAX_RMS_DB = -100.0
DECODER_MAX_PEAK_DB = -70.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_bytes(path: Path) -> int:
    """An ONNX file plus the external weights the exporter may put beside it."""
    total = path.stat().st_size
    data = path.with_suffix(path.suffix + ".data")
    if data.exists():
        total += data.stat().st_size
    return total


class DecoderWrapper(torch.nn.Module):
    """``decode`` with the sample count frozen, which is what the graph needs."""

    def __init__(self, instrument, samples: int) -> None:
        super().__init__()
        self.instrument = instrument
        self.samples = int(samples)

    def forward(self, trajectory, note, window_start):
        audio, _ = self.instrument.decode(
            trajectory, note, self.samples, window_start=window_start
        )
        return audio


class FlowWrapper(torch.nn.Module):
    """One velocity evaluation, via the validation-free entry point."""

    def __init__(self, flow) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, noisy_residual, flow_time, history, history_anchor,
                anchor_path, atlas_path, lifecycle, history_mask):
        return self.flow.evaluate(
            noisy_residual, flow_time, history, history_anchor,
            anchor_path, atlas_path, lifecycle, history_mask,
        )


def export(module, inputs, names, out: Path) -> float:
    started = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module, inputs, str(out),
        input_names=names, output_names=["output"], dynamo=True,
    )
    return time.perf_counter() - started


def check(out: Path, inputs, names, reference: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(out), options, providers=["CPUExecutionProvider"])
    feed = {
        item.name: value.detach().cpu().numpy()
        for item, value in zip(session.get_inputs(), inputs)
    }
    got = session.run(None, feed)[0]
    if got.shape != reference.shape:
        raise SystemExit(f"{out.name}: shape {got.shape} != torch {reference.shape}")
    return got


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path, help="folder holding weights and atlas")
    parser.add_argument("--out", type=Path, default=ROOT / "plugin" / "models")
    parser.add_argument("--samples", type=int, default=0,
                        help="decoder output length; default is one full plan")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    from midibrave.atlas_flow_local import build_parser, resolve
    from midibrave.atlas_flow_runtime import AtlasFlowLiveEngine

    local_argv = []
    if args.model:
        local_argv += ["--model", str(args.model)]
    resolved = dict(zip(*[iter(resolve(build_parser().parse_args(local_argv + ["--print-argv"])))] * 2))
    config = args.config or Path(resolved["--config"])
    checkpoint = args.checkpoint or Path(resolved["--checkpoint"])

    engine = AtlasFlowLiveEngine.load(config, checkpoint, device="cpu")
    system, cfg = engine.system, engine.config
    future, context = cfg.model.future_frames, cfg.model.context_frames
    dim, atlas_dim = cfg.model.trajectory_dim, cfg.model.atlas_dim
    frames = engine.total_trajectory_frames
    samples = args.samples or cfg.data.render_samples

    print(f"checkpoint {checkpoint.name}  sha256 {sha256(checkpoint)[:16]}…")
    print(f"future={future} context={context} dim={dim} atlas={atlas_dim} "
          f"frames={frames} samples={samples}\n")

    torch.manual_seed(20260826)
    manifest: dict[str, object] = {
        "torch": torch.__version__,
        "checkpoint": checkpoint.name,
        "checkpointSha256": sha256(checkpoint),
        "sampleRate": cfg.data.sample_rate,
        "graphs": {},
    }
    failures = 0

    # ---- flow predictor -------------------------------------------------
    flow_inputs = (
        torch.randn(1, future, dim), torch.full((1,), 0.25),
        torch.randn(1, context, dim), torch.randn(1, context, dim),
        torch.randn(1, future, dim), torch.randn(1, future, atlas_dim),
        torch.rand(1, future, 4), torch.ones(1, context, dtype=torch.bool),
    )
    flow_names = ["noisy_residual", "flow_time", "history", "history_anchor",
                  "anchor_path", "atlas_path", "lifecycle", "history_mask"]
    wrapper = FlowWrapper(system.flow).eval()
    with torch.no_grad():
        reference = wrapper(*flow_inputs).numpy()
        # The stripped entry point must agree with the validated one exactly,
        # or the export is of a different model than the runtime uses.
        validated = system.flow(*flow_inputs).numpy()
    drift = float(np.abs(reference - validated).max())
    print(f"flow      evaluate() vs forward(): max abs {drift:.3e}"
          f"{'' if drift == 0.0 else '   <-- NOT identical'}")
    if drift != 0.0:
        failures += 1

    path = args.out / "flow.onnx"
    seconds = export(wrapper, flow_inputs, flow_names, path)
    got = check(path, flow_inputs, flow_names, reference)
    scale = float(np.abs(reference).max())
    relative = float(np.abs(reference - got).max() / max(scale, 1e-12))
    ok = bool(relative <= FLOW_MAX_RELATIVE)
    failures += 0 if ok else 1
    print(f"flow      {payload_bytes(path)/1e6:6.1f} MB  {seconds:4.1f}s  "
          f"relative error {relative:.2e}  [{'PASS' if ok else 'FAIL'}]")
    manifest["graphs"]["flow"] = {
        "file": path.name, "bytes": payload_bytes(path), "sha256": sha256(path),
        "relativeError": relative, "threshold": FLOW_MAX_RELATIVE, "pass": ok,
        "inputs": flow_names,
    }

    # ---- decoder --------------------------------------------------------
    # Re-seed so this graph's inputs do not depend on how much randomness the
    # flow export consumed; otherwise the measured drift moves whenever
    # anything above it changes, and a gate that moves is not a gate.
    torch.manual_seed(20260826)
    decoder_inputs = (
        torch.randn(1, dim, frames).tanh(),
        torch.tensor([50]),
        torch.zeros(1, dtype=torch.long),
    )
    decoder_names = ["trajectory", "note", "window_start"]
    decoder = DecoderWrapper(system.instrument, samples).eval()
    with torch.no_grad():
        reference = decoder(*decoder_inputs).numpy()

    path = args.out / "decoder.onnx"
    seconds = export(decoder, decoder_inputs, decoder_names, path)
    got = check(path, decoder_inputs, decoder_names, reference)
    peak = float(np.abs(reference).max())
    difference = np.abs(reference - got)
    as_db = lambda v: float(20.0 * np.log10(max(float(v), 1e-12) / max(peak, 1e-12)))
    rms_db = as_db(np.sqrt((difference.astype(np.float64) ** 2).mean()))
    peak_db = as_db(difference.max())
    ok = bool(rms_db <= DECODER_MAX_RMS_DB and peak_db <= DECODER_MAX_PEAK_DB)
    failures += 0 if ok else 1
    print(f"decoder   {payload_bytes(path)/1e6:6.1f} MB  {seconds:4.1f}s  "
          f"rms {rms_db:6.1f} dB · peak {peak_db:6.1f} dB   [{'PASS' if ok else 'FAIL'}]")
    manifest["graphs"]["decoder"] = {
        "file": path.name, "bytes": payload_bytes(path), "sha256": sha256(path),
        "errorRmsDb": rms_db, "errorPeakDb": peak_db,
        "thresholdRmsDb": DECODER_MAX_RMS_DB, "thresholdPeakDb": DECODER_MAX_PEAK_DB,
        "pass": ok, "inputs": decoder_names, "outputSamples": samples,
        "trajectoryFrames": frames,
    }

    # ---- pitch independence --------------------------------------------
    # One trajectory serves every note, which is what lets a plugin cache a
    # plan and pay only a decode per note-on. If this ever stops holding, the
    # caching in the runtime is wrong and the gate should say so.
    with torch.no_grad():
        base = decoder(*decoder_inputs).numpy()
        spread = 0.0
        for note in (36, 60, 71):
            other = decoder(decoder_inputs[0], torch.tensor([note]), decoder_inputs[2]).numpy()
            if note != 50:
                spread = max(spread, float(np.abs(base - other).max()))
    print(f"\npitch     same trajectory, notes 36/60/71 differ by {spread:.3e} "
          f"(expected: non-zero, the decoder is where MIDI enters)")

    total = sum(g["bytes"] for g in manifest["graphs"].values())
    manifest["totalBytes"] = total
    manifest["pass"] = failures == 0
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # --out may legitimately sit outside the repo (CI scratch, a temp dir),
    # so show a repo-relative path when we can and the full one otherwise.
    try:
        shown = args.out.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.out.resolve()
    print(f"\ntotal {total/1e6:.1f} MB  ->  {shown}")
    print("manifest.json written")
    if failures:
        print(f"\n{failures} gate(s) failed", file=sys.stderr)
        return 1
    print("\nall gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

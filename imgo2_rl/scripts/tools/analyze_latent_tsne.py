#!/usr/bin/env python3
"""Reproducible t-SNE views of towing latent; requires numpy, sklearn, matplotlib.

--report reads case/phase means from play_towing_test.py --dump-latent.
--samples reads a JSON list (or {"samples": [...]}) of short-window records:
  {"latent": [16 floats], "case": {...}, "local_slope_deg": ..., "segment": ...}
Labels only color an embedding fitted without labels. This is visualization,
not clustering or evidence of physical parameter identification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path

GROUPS = ("connection", "cart_mass_kg", "grade_deg", "velocity_mps",
          "local_slope_deg", "segment")


def collect(payload, phase, sample_mode=False, include_nocart=False):
    entries = payload if isinstance(payload, list) else payload.get(
        "samples" if sample_mode else "cases", [])
    vectors, records = [], []
    skipped = {"missing_latent": 0, "invalid_latent": 0, "nocart": 0}
    for index, entry in enumerate(entries):
        metrics = entry.get("metrics", entry)
        case = metrics.get("case") or entry.get("case") or {}
        if case.get("cart_present") is False and not include_nocart:
            skipped["nocart"] += 1
            continue
        vector = entry.get("latent") if sample_mode else (metrics.get("latent") or {}).get(phase)
        if vector is None or vector == []:
            skipped["missing_latent"] += 1
            continue
        try:
            vector = [float(x) for x in vector]
            valid = len(vector) == 16 and all(math.isfinite(x) for x in vector)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            skipped["invalid_latent"] += 1
            continue
        record = {"source_index": index, "case": case,
                  "labels": {key: entry.get(key, case.get(key)) for key in GROUPS}}
        vectors.append(vector)
        records.append(record)
    return vectors, records, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path)
    source.add_argument("--samples", type=Path)
    parser.add_argument("--phase", choices=("station", "tow", "coast"), default="tow")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perplexities", type=float, nargs="+", default=[10, 30, 50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--include-nocart", action="store_true")
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=JSON_VALUE",
                        help='Fixed-condition subset, e.g. --filter velocity_mps=1.0')
    args = parser.parse_args(argv)
    if args.iterations < 250 or any(p <= 0 or not math.isfinite(p) for p in args.perplexities) or any(s < 0 for s in args.seeds):
        parser.error("iterations must be >=250; perplexities must be positive and finite; seeds >=0")
    path = args.report or args.samples
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
        vectors, records, skipped = collect(payload, args.phase, bool(args.samples), args.include_nocart)
        filters = {}
        for item in args.filter:
            key, value = item.split("=", 1)
            filters[key] = json.loads(value)
        selected = [(v, r) for v, r in zip(vectors, records)
                    if all(r["labels"].get(k, r["case"].get(k)) == val for k, val in filters.items())]
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        parser.error(str(exc))
    if len(selected) < 5:
        parser.error(f"Only {len(selected)} valid selected latent vectors; need >=5. "
                     "Use a report produced with --dump-latent or short-window --samples. "
                     f"Skipped: {skipped}")
    perplexities = list(dict.fromkeys(p for p in args.perplexities if p < len(selected)))
    if not perplexities:
        parser.error("No perplexity smaller than selected sample count; set --perplexities")
    try:
        import numpy as np
        import sklearn
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE, trustworthiness
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        parser.error(f"Missing dependency: {exc}. Install numpy scikit-learn matplotlib.")
    vectors, records = zip(*selected)
    scaler = StandardScaler()
    x = scaler.fit_transform(np.asarray(vectors, dtype=float))
    if not np.any(np.var(x, axis=0) > 0):
        parser.error("All latent vectors are identical; t-SNE would be uninformative")
    args.output.mkdir(parents=True, exist_ok=True)
    active = [key for key in GROUPS if any(r["labels"][key] is not None for r in records)]
    if not active:
        active = ["connection"]  # explicitly show missing labels
    runs = []
    for perplexity in perplexities:
        for seed in dict.fromkeys(args.seeds):
            kwargs = dict(n_components=2, perplexity=perplexity, random_state=seed,
                          init="random", learning_rate="auto", method="barnes_hut")
            # scikit-learn renamed n_iter to max_iter in 1.5.
            kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = args.iterations
            model = TSNE(**kwargs)
            xy = model.fit_transform(x)
            tag = f"tsne_{args.phase if args.report else 'windows'}_p{perplexity:g}_s{seed}"
            fig, axes = plt.subplots(2, math.ceil(len(active) / 2), figsize=(6 * math.ceil(len(active)/2), 9), squeeze=False)
            for ax, key in zip(axes.flat, active):
                values = [r["labels"][key] for r in records]
                labels = sorted(set(values), key=lambda v: (0, v) if isinstance(v, (int, float)) else (1, str(v)))
                for i, label in enumerate(labels):
                    mask = np.asarray([v == label for v in values])
                    ax.scatter(xy[mask, 0], xy[mask, 1], s=13, alpha=.7,
                               color=plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")(i % (10 if len(labels) <= 10 else 20)),
                               label=f"{label if label is not None else 'missing'} (n={sum(mask)})")
                ax.set_title(key + (" (lane maximum, not local slope)" if key == "grade_deg" else ""))
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")
                ax.legend(fontsize=8)
            for ax in list(axes.flat)[len(active):]:
                ax.set_visible(False)
            synthetic = isinstance(payload, dict) and payload.get("synthetic", False)
            fig.suptitle(("SYNTHETIC TOOL CHECK - NOT EXPERIMENT DATA\n" if synthetic else "")
                         + f"Towing latent | n={len(x)} | perplexity={perplexity:g} | seed={seed}\n"
                         + (f"{args.phase} phase means" if args.report else "short-window samples"))
            fig.tight_layout(rect=(0, 0, 1, .93))
            fig.savefig(args.output / f"{tag}.png", dpi=200)
            fig.savefig(args.output / f"{tag}.pdf")
            plt.close(fig)
            with (args.output / f"{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")  # 显式 LF，见 .gitattributes
                writer.writerow(["source_index", "tsne_1", "tsne_2", *GROUPS])
                for r, point in zip(records, xy):
                    writer.writerow([r["source_index"], *point, *(r["labels"][k] for k in GROUPS)])
            runs.append({"tag": tag, "perplexity": perplexity, "seed": seed,
                         "kl_divergence": float(model.kl_divergence_),
                         "trustworthiness": float(trustworthiness(x, xy, n_neighbors=min(5, (len(x)-1)//2)))})
            print(f"Saved {tag}: KL={model.kl_divergence_:.4f}", flush=True)
    metadata = {"source": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                "sample_type": "phase_mean" if args.report else "window", "phase": args.phase if args.report else None,
                "samples": len(x), "skipped": skipped, "filters": filters,
                "include_nocart": args.include_nocart, "iterations": args.iterations,
                "standardization_mean": scaler.mean_.tolist(), "standardization_scale": scaler.scale_.tolist(),
                "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
                "runs": runs, "skipped_perplexities": [p for p in args.perplexities if p >= len(x)],
                "caution": "Labels only color the same embedding. t-SNE is visualization, not clustering. "
                           "Phase means cannot resolve local up/down slope; grade_deg is lane maximum. "
                           "Trustworthiness measures neighborhood preservation, not label separability."}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Recalculate reference_ranges.json from pairs of genuine texts.

Put two genuine texts per person in data/genuine_pairs/ as personX_a.txt and
personX_b.txt (ideally a supervised sample and a remote response written by
the same person). For each feature this script measures the absolute
difference within every pair and sets:

    typical = 75th percentile of within-person differences
    high    = 95th percentile of within-person differences

With few pairs these percentiles are unstable, so they are shrunk towards the
provisional defaults: weight = n / (n + PRIOR_STRENGTH). Ranges are marked
"calibrated" only once at least --min-pairs pairs have been used.

Usage:
    python calibrate.py              # show the proposed ranges (no changes)
    python calibrate.py --write      # save them to src/reference_ranges.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from src.compare import FEATURE_SPECS, REFERENCE_PATH, feature_differences, load_reference_ranges
from src.features import extract_features

PAIRS_DIR = Path(__file__).parent / "data" / "genuine_pairs"
PRIOR_STRENGTH = 10
MIN_RATIO = 1.25  # keep "high" meaningfully above "typical"


def percentile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def load_pairs(pairs_dir: Path) -> list[tuple[str, str, str]]:
    pairs = []
    for a_path in sorted(pairs_dir.glob("*_a.txt")):
        person = re.sub(r"_a$", "", a_path.stem)
        b_path = a_path.with_name(f"{person}_b.txt")
        if b_path.exists():
            pairs.append((person, a_path.read_text(encoding="utf-8"), b_path.read_text(encoding="utf-8")))
        else:
            print(f"  skipping {a_path.name}: no matching {b_path.name}")
    return pairs


def calibrate(pairs, prior: dict, min_pairs: int) -> dict:
    diffs_by_feature: dict[str, list[float]] = {s.key: [] for s in FEATURE_SPECS}
    for _, text_a, text_b in pairs:
        diffs = feature_differences(extract_features(text_a), extract_features(text_b))
        for key, value in diffs.items():
            if value is not None:
                diffs_by_feature[key].append(value)

    out = {"_meta": {
        "status": "calibrated" if len(pairs) >= min_pairs else "provisional",
        "description": (f"Calibrated by calibrate.py on {len(pairs)} genuine pairs "
                        f"(75th/95th percentiles, shrunk towards defaults with prior strength {PRIOR_STRENGTH})."),
        "n_pairs": len(pairs),
        "source": "calibrate.py",
        "date": date.today().isoformat(),
        "prior": {s.key: prior[s.key] for s in FEATURE_SPECS},
    }}
    for spec in FEATURE_SPECS:
        values = diffs_by_feature[spec.key]
        p = prior[spec.key]
        if not values:
            out[spec.key] = dict(p)
            continue
        w = len(values) / (len(values) + PRIOR_STRENGTH)
        typical = w * percentile(values, 0.75) + (1 - w) * p["typical"]
        high = w * percentile(values, 0.95) + (1 - w) * p["high"]
        high = max(high, typical * MIN_RATIO)
        out[spec.key] = {"typical": round(typical, 4), "high": round(high, 4), "n": len(values)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-dir", type=Path, default=PAIRS_DIR)
    parser.add_argument("--output", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--min-pairs", type=int, default=30,
                        help="pairs needed before ranges are marked calibrated (default 30)")
    parser.add_argument("--write", action="store_true", help="save the new ranges")
    args = parser.parse_args()

    current = load_reference_ranges(args.output)
    # Always shrink towards the original defaults, not a previous calibration.
    prior = current.get("_meta", {}).get("prior") or current

    pairs = load_pairs(args.pairs_dir)
    if not pairs:
        print(f"No genuine pairs found in {args.pairs_dir}")
        return 1
    print(f"Found {len(pairs)} genuine pair(s): {', '.join(p[0] for p in pairs)}\n")

    new = calibrate(pairs, prior, args.min_pairs)
    print(f"{'Feature':<26}{'old typical':>12}{'old high':>10}{'new typical':>13}{'new high':>10}")
    for spec in FEATURE_SPECS:
        o, n = current[spec.key], new[spec.key]
        print(f"{spec.key:<26}{o['typical']:>12.3f}{o['high']:>10.3f}{n['typical']:>13.3f}{n['high']:>10.3f}")
    print(f"\nStatus: {new['_meta']['status']} ({len(pairs)} pairs; {args.min_pairs} needed for 'calibrated')")

    if args.write:
        args.output.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
        print(f"Saved to {args.output}")
    else:
        print("Dry run: nothing saved. Use --write to save.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

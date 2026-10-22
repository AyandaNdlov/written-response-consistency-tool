"""Run the three evaluation conditions on the bundled sample texts.

    genuine-genuine, genuine-AI, genuine-AI-edited

and print how many features fall in each band, plus every genuine pair in
data/genuine_pairs/ (the false-flag check). Usage: python evaluate.py [-v]
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.compare import AMBER, RED, compare_texts, format_value, load_reference_ranges
from src.paragraphs import analyse_paragraphs

ROOT = Path(__file__).parent
SAMPLES = ROOT / "data" / "samples"
# Each condition pairs the same supervised baseline with a different remote text.
CONDITIONS = {
    "genuine-genuine": "remote_genuine_person1.txt",
    "genuine-AI": "remote_ai_person1.txt",
    "genuine-AI-edited": "remote_ai_edited_person1.txt",
}


def describe(name: str, baseline: str, remote: str, verbose: bool) -> dict:
    """Compare one pair, print the summary and flagged features, and return the result."""
    result = compare_texts(baseline, remote)
    print(f"\n== {name} ==")
    if result["length_status"] == "refuse":
        print("  refused:", result["length_message"])
        return result
    print("  " + result["summary"])
    flagged = [r for r in result["rows"] if r["band"] in (AMBER, RED)]
    for r in flagged:
        print(f"    {r['band']:<20} {r['label']:<34} diff {format_value(r['difference'], r['decimals'])}"
              f" (typical {r['typical']}, high {r['high']})")
    if verbose:
        paras = analyse_paragraphs(remote, result["baseline_features"], load_reference_ranges())
        for p in paras:
            if p["eligible"]:
                print(f"    para {p['index']}: score {p['score']:.2f} {p['band']}")
    return result


def main() -> None:
    """Run the three sample conditions, then the false-flag check on genuine pairs."""
    verbose = "-v" in sys.argv
    baseline = (SAMPLES / "baseline_person1.txt").read_text(encoding="utf-8")
    for name, fname in CONDITIONS.items():
        describe(name, baseline, (SAMPLES / fname).read_text(encoding="utf-8"), verbose)

    # Genuine pairs are by the same person, so anything outside typical is a false flag.
    print("\n== Genuine pairs (false-flag check) ==")
    red_total = outside_total = assessed_total = 0
    for a_path in sorted((ROOT / "data" / "genuine_pairs").glob("*_a.txt")):
        b_path = a_path.with_name(a_path.name.replace("_a.txt", "_b.txt"))
        res = describe(a_path.stem[:-2], a_path.read_text(encoding="utf-8"),
                       b_path.read_text(encoding="utf-8"), False)
        if res["length_status"] != "refuse":
            red_total += res["n_red"]
            outside_total += res["n_outside"]
            assessed_total += res["n_assessed"]
    if assessed_total:
        print(f"\nFalse-flag rate on genuine pairs: {outside_total}/{assessed_total} features outside "
              f"typical ({100 * outside_total / assessed_total:.0f}%), {red_total} red")


if __name__ == "__main__":
    main()

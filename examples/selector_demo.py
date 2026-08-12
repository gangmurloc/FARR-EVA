"""Load the frozen selector and score one synthetic candidate group.

This checks the artifact/API boundary only. The fabricated features are not a
benchmark and the selected answer has no scientific interpretation.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from farr_star.eva_selector import EvidenceVerifiedAbstainingSelector


def main() -> None:
    selector, metadata = EvidenceVerifiedAbstainingSelector.load(
        str(ROOT / "artifacts" / "farr_eva_v1.joblib")
    )
    rows = []
    for method, entailment, margin in (
        ("flare-embedded", 0.72, 0.32),
        ("ircot", 0.84, 0.51),
        ("farr", 0.65, 0.21),
    ):
        features = {name: 0.0 for name in selector.feature_names}
        features["answer_entail_max"] = entailment
        features["answer_margin_max"] = margin
        features["proof_margin_min"] = margin
        rows.append({"method": method, "features": features})

    selected, probability, utilities, switched = selector.choose(rows)
    print(f"artifact schema: {metadata.get('schema')}")
    print(f"feature count: {len(selector.feature_names)}")
    print(f"selected: {selected}")
    print(f"switch probability: {probability:.6f}")
    print(f"switched from FARR anchor: {switched}")
    print("utilities:")
    for method, value in utilities.items():
        print(f"  {method}: {value:.6f}")


if __name__ == "__main__":
    main()

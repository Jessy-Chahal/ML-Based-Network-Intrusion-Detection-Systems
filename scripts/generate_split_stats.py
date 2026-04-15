"""
Reads .npz and _label_map.npy files and produces data/splits/split_stats.json.
"""
import json
import numpy as np
from pathlib import Path

SPLITS_DIR = Path("./data/splits")

IMBALANCE_WARNINGS = {
    "cicids2017": [
        "BENIGN is 80.3% - use class_weight='balanced' in all models",
        "Heartbleed(11), SQLi(21), Infiltration(36) too sparse for reliable evaluation",
        "PortScan, Bot, DoS Hulk have known labeling/stat errors - interpret carefully",
        "Flow Duration has min=-13 (CICFlowMeter artifact) - clip to 0 in attack design",
    ],
    "nslkdd": [
        "R2L=0.79%, U2R=0.04% - report per-class F1, not overall accuracy",
        "DoS(36.46%) and Probe(9.25%) are the reliable attack targets",
        "num_outbound_cmds confirmed constant and dropped (42 → 41 → 38 features after encoding)",
    ],
    "unswnb15": [
        "Normal is 80% - use class_weight='balanced' in all models",
        "Worms(0.05%), Fuzzers(0.09%), Analysis(0.10%) too sparse after dedup",
        "31.64% duplicates removed (141,742 rows)",
    ],
}

RECOMMENDED_ATTACK_TARGETS = {
    "cicids2017": ["DoS Hulk", "PortScan", "DDoS"],
    "nslkdd": ["DoS", "Probe"],
    "unswnb15": ["DoS", "Exploits", "Reconnaissance"],
}


def class_dist(y, label_map):
    ids, counts = np.unique(y, return_counts=True)
    return {label_map.get(int(i), str(i)): int(c) for i, c in zip(ids, counts)}


def main():
    stats = {}
    for name in ["cicids2017", "nslkdd", "unswnb15"]:
        npz_path  = SPLITS_DIR / f"{name}.npz"
        lmap_path = SPLITS_DIR / f"{name}_label_map.npy"

        if not npz_path.exists():
            print(f"WARNING: {npz_path} not found, skipping")
            continue

        data = np.load(npz_path,  allow_pickle=True)
        lmap = np.load(lmap_path, allow_pickle=True).item()

        total      = sum(len(data[f"y_{s}"]) for s in ["train", "val", "test"])
        n_features = int(data["X_train"].shape[1])
        feat_names = data["feature_names"].tolist() if "feature_names" in data.files else []

        dist = {s: class_dist(data[f"y_{s}"], lmap) for s in ["train", "val", "test"]}

        train_counts    = list(dist["train"].values())
        imbalance_ratio = round(max(train_counts) / max(min(train_counts), 1), 1)

        stats[name] = {
            "total_rows": total,
            "n_features": n_features,
            "feature_names": feat_names,
            "split_sizes": {s: int(len(data[f"y_{s}"])) for s in ["train", "val", "test"]},
            "class_distribution": dist,
            "label_map": {str(k): v for k, v in lmap.items()},
            "imbalance_ratio_train": imbalance_ratio,
            "imbalance_warnings": IMBALANCE_WARNINGS.get(name, []),
            "recommended_attack_targets": RECOMMENDED_ATTACK_TARGETS.get(name, []),
        }

        print(f"\n{name.upper()} | {total:,} rows | {n_features} features | imbalance: {imbalance_ratio}x")
        for split, d in dist.items():
            print(f"  {split}: {d}")

    out = SPLITS_DIR / "split_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()

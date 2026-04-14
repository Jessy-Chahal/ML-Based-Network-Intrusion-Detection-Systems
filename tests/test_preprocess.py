"""
Validates .npz splits produced by preprocess.py.
Run with: pytest tests/test_preprocess.py -v
"""
import json
import numpy as np
import pytest
from pathlib import Path

SPLITS_DIR = Path("data/splits")
DATASETS   = ["cicids2017", "nslkdd", "unswnb15"]


@pytest.fixture(params=DATASETS)
def dataset(request):
    name = request.param
    data = np.load(SPLITS_DIR / f"{name}.npz",           allow_pickle=True)
    lmap = np.load(SPLITS_DIR / f"{name}_label_map.npy",  allow_pickle=True).item()
    return name, data, lmap


# ── File existence ─────────────────────────────────────────────────────────────

def test_npz_files_exist():
    for name in DATASETS:
        assert (SPLITS_DIR / f"{name}.npz").exists(), f"{name}.npz missing"

def test_label_map_files_exist():
    for name in DATASETS:
        assert (SPLITS_DIR / f"{name}_label_map.npy").exists(), \
            f"{name}_label_map.npy missing"

def test_split_stats_json_exists():
    assert (SPLITS_DIR / "split_stats.json").exists(), \
        "split_stats.json missing - run scripts/generate_split_stats.py"


# ── Required keys ──────────────────────────────────────────────────────────────

def test_required_keys_present(dataset):
    name, data, _ = dataset
    for key in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        assert key in data.files, f"{key} missing from {name}.npz"


# ── Shape consistency ──────────────────────────────────────────────────────────

def test_feature_count_consistent_across_splits(dataset):
    name, data, _ = dataset
    n = data["X_train"].shape[1]
    assert data["X_val"].shape[1]  == n, f"{name}: val feature count mismatch"
    assert data["X_test"].shape[1] == n, f"{name}: test feature count mismatch"

def test_X_y_length_match(dataset):
    name, data, _ = dataset
    for split in ["train", "val", "test"]:
        assert len(data[f"X_{split}"]) == len(data[f"y_{split}"]), \
            f"{name}: X/y length mismatch in {split}"

def test_split_ratio_approximately_70_15_15(dataset):
    name, data, _ = dataset
    total     = sum(len(data[f"y_{s}"]) for s in ["train", "val", "test"])
    train_pct = len(data["y_train"]) / total
    val_pct   = len(data["y_val"])   / total
    test_pct  = len(data["y_test"])  / total
    assert 0.68 <= train_pct <= 0.72, f"{name}: train={train_pct:.2%} (expected ~70%)"
    assert 0.13 <= val_pct   <= 0.17, f"{name}: val={val_pct:.2%} (expected ~15%)"
    assert 0.13 <= test_pct  <= 0.17, f"{name}: test={test_pct:.2%} (expected ~15%)"


# ── Data quality ───────────────────────────────────────────────────────────────

def test_no_nan_in_X(dataset):
    name, data, _ = dataset
    for split in ["train", "val", "test"]:
        assert not np.isnan(data[f"X_{split}"]).any(), \
            f"{name}: NaN found in X_{split}"

def test_no_inf_in_X(dataset):
    name, data, _ = dataset
    for split in ["train", "val", "test"]:
        assert not np.isinf(data[f"X_{split}"]).any(), \
            f"{name}: Inf found in X_{split}"

def test_dtype_is_float32(dataset):
    name, data, _ = dataset
    for split in ["train", "val", "test"]:
        assert data[f"X_{split}"].dtype == np.float32, \
            f"{name}: X_{split} dtype={data[f'X_{split}'].dtype}, expected float32"


# ── Label integrity ────────────────────────────────────────────────────────────

def test_label_ids_match_label_map(dataset):
    name, data, lmap = dataset
    all_y      = np.concatenate([data[f"y_{s}"] for s in ["train", "val", "test"]])
    unique_ids = set(np.unique(all_y).tolist())
    map_keys   = set(lmap.keys())
    assert unique_ids == map_keys, \
        f"{name}: label IDs {unique_ids} != map keys {map_keys}"

def test_stratification_preserved(dataset):
    name, data, lmap = dataset
    def dist(y):
        ids, counts = np.unique(y, return_counts=True)
        return {int(i): c / len(y) for i, c in zip(ids, counts)}
    train_d = dist(data["y_train"])
    for split in ["val", "test"]:
        for cls, prop in dist(data[f"y_{split}"]).items():
            if cls not in train_d:
                continue
            diff = abs(train_d[cls] - prop)
            assert diff < 0.05, \
                f"{name}: class '{lmap.get(cls, cls)}' differs by {diff:.2%} in {split}"


# ── Sanity bounds ──────────────────────────────────────────────────────────────

def test_minimum_feature_count(dataset):
    name, data, _ = dataset
    n = data["X_train"].shape[1]
    assert n >= 20, f"{name}: only {n} features after cleaning - unexpectedly low"

def test_minimum_training_samples(dataset):
    name, data, _ = dataset
    n = len(data["y_train"])
    assert n >= 1000, f"{name}: only {n} training samples"


# ── split_stats.json integrity ─────────────────────────────────────────────────

def test_split_stats_covers_all_datasets():
    stats = json.loads((SPLITS_DIR / "split_stats.json").read_text())
    for name in DATASETS:
        assert name in stats, f"{name} missing from split_stats.json"

def test_split_stats_row_counts_match_npz():
    stats = json.loads((SPLITS_DIR / "split_stats.json").read_text())
    for name in DATASETS:
        data   = np.load(SPLITS_DIR / f"{name}.npz", allow_pickle=True)
        actual = sum(len(data[f"y_{s}"]) for s in ["train", "val", "test"])
        assert actual == stats[name]["total_rows"], \
            f"{name}: stats={stats[name]['total_rows']} but npz={actual}"

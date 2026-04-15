"""
Injects feature_names into existing .npz files produced in preprocess.py.
Run once after preprocess.py has been executed.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SPLITS_DIR = Path("./data/splits")


def make_fallback_feature_names(dataset, n_features):
    return [f"{dataset}_feature_{i:03d}" for i in range(n_features)]


def get_cicids_feature_names(expected_n):
    data_dir = Path("data/CICIDS2017/MachineLearningCVE")
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print("  WARNING: CICIDS2017 raw CSVs not found; using fallback feature names")
        return make_fallback_feature_names("cicids2017", expected_n)

    first_csv = csv_files[0]
    sample = pd.read_csv(first_csv, nrows=1000, encoding='latin-1', low_memory=False)
    sample.columns = sample.columns.str.strip()
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    constant = [c for c in numeric_cols if sample[c].nunique() <= 1]
    feature_names = [c for c in numeric_cols if c not in constant]

    if len(feature_names) != expected_n:
        print(f"  WARNING: CICIDS2017 derived {len(feature_names)} names but expected {expected_n}; "
              "using fallback names to avoid misalignment")
        return make_fallback_feature_names("cicids2017", expected_n)

    print(f"  CICIDS2017: derived {len(feature_names)} feature names from {first_csv.name}")
    return feature_names


def get_nslkdd_feature_names(expected_n):
    all_cols = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    ]
    # num_outbound_cmds confirmed constant - dropped by preprocess.py
    feature_names = [c for c in all_cols if c != "num_outbound_cmds"]

    if len(feature_names) != expected_n:
        print(f"  WARNING: NSL-KDD derived {len(feature_names)} names but expected {expected_n}; "
              "using fallback names to avoid misalignment")
        return make_fallback_feature_names("nslkdd", expected_n)

    print(f"  NSL-KDD: derived {len(feature_names)} feature names")
    return feature_names


def get_unswnb15_feature_names(expected_n):
    data_csv = Path("data/UNSW-NB15/Data.csv")
    if not data_csv.exists():
        print("  WARNING: UNSW-NB15 raw Data.csv not found; using fallback feature names")
        return make_fallback_feature_names("unswnb15", expected_n)

    sample = pd.read_csv(data_csv, nrows=1000, low_memory=False)
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Label']
    constant = [c for c in numeric_cols if sample[c].nunique() <= 1]
    feature_names = [c for c in numeric_cols if c not in constant]

    if len(feature_names) != expected_n:
        print(f"  WARNING: UNSW-NB15 derived {len(feature_names)} names but expected {expected_n}; "
              "using fallback names to avoid misalignment")
        return make_fallback_feature_names("unswnb15", expected_n)

    print(f"  UNSW-NB15: derived {len(feature_names)} feature names from {data_csv}")
    return feature_names


def patch(name, feature_name_getter):
    path = SPLITS_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    with np.load(path, allow_pickle=True) as data:
        n_in_array = data["X_train"].shape[1]
        payload = {k: data[k] for k in data.files if k != "feature_names"}

    feature_names = feature_name_getter(n_in_array)

    if len(feature_names) != n_in_array:
        raise ValueError(
            f"{name}: feature_names length {len(feature_names)} does not match X_train columns {n_in_array}"
        )
    print(f"  ✓ {name}: feature count matches ({n_in_array})")

    payload["feature_names"] = np.array(feature_names)
    np.savez_compressed(path, **payload)
    print(f"  Saved {path}")


if __name__ == "__main__":
    print("=== Patching feature_names into .npz files ===\n")
    print("CICIDS2017")
    patch("cicids2017", get_cicids_feature_names)
    print("\nNSL-KDD")
    patch("nslkdd", get_nslkdd_feature_names)
    print("\nUNSW-NB15")
    patch("unswnb15", get_unswnb15_feature_names)
    print("\nDone.")

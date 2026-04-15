import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

### Output Directory ###
SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


### Shared Utilities ###

def drop_constant_features(df, numeric_cols):
    constant = [col for col in numeric_cols if df[col].nunique() <= 1]
    print(f"  Dropping {len(constant)} constant features: {constant}")
    return df.drop(columns=constant), constant


def handle_nulls_and_inf(df, numeric_cols):
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    null_counts = df[numeric_cols].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols):
        print(f"  Imputing NaN in {len(null_cols)} columns with column mean")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df


def save_splits(X, y, name):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    out = SPLITS_DIR / name
    np.savez_compressed(
        out,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test
    )
    print(f"  Saved {out}.npz")
    print(f"    Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    label_ids, label_counts = np.unique(y_train, return_counts=True)
    print(f"    Label distribution (train): {dict(zip(label_ids.tolist(), label_counts.tolist()))}")


### CICIDS2017 ###

def process_cicids2017():
    print("\n=== CICIDS2017 ===")
    data_dir = Path("data/CICIDS2017/MachineLearningCVE")

    files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]

    dfs = []
    for f in files:
        part = pd.read_csv(data_dir / f, encoding='utf-8', low_memory=False)
        print(f"  Loaded {f}: {part.shape}")
        dfs.append(part)

    df = pd.concat(dfs, ignore_index=True)

    # Strip leading/trailing whitespace from column names - known CICIDS2017 issue
    df.columns = df.columns.str.strip()
    print(f"  Total after concat: {df.shape}")

    # Drop duplicates - CICFlowMeter artifacts, not real repeated flows
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {before - len(df)} duplicate rows")

    # Replace infinite values and impute nulls
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df, _ = drop_constant_features(df, numeric_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = handle_nulls_and_inf(df, numeric_cols)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['Label'])
    print(f"  Classes: {list(le.classes_)}")

    label_map = {i: c for i, c in enumerate(le.classes_)}
    np.save(SPLITS_DIR / "cicids2017_label_map.npy", label_map)

    X = df[numeric_cols].values.astype(np.float32)
    save_splits(X, y, "cicids2017")


### NSL-KDD ###

def process_nslkdd():
    print("\n=== NSL-KDD ===")

    col_names = [
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
        "label", "difficulty"
    ]

    train = pd.read_csv(
        "data/NSL-KDD/KDDTrain+.txt", header=None, names=col_names
    )
    test = pd.read_csv(
        "data/NSL-KDD/KDDTest+.txt", header=None, names=col_names
    )

    df = pd.concat([train, test], ignore_index=True)
    print(f"  Loaded: {df.shape}")

    # Drop difficulty column - not a feature
    df = df.drop(columns=['difficulty'])

    # Encode categorical features
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Map specific attack labels to 5-class taxonomy
    attack_map = {
        'normal': 'Normal',
        # DoS
        'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
        'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
        'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
        # Probe
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
        'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
        # R2L
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
        'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
        'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L',
        'named': 'R2L', 'snmpattack': 'R2L', 'snmpgetattack': 'R2L',
        'worm': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
        # U2R
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
        'rootkit': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
    }

    df['category'] = df['label'].map(attack_map)

    # Any unmapped labels (rare variants not in training set) default to R2L
    unmapped = df['category'].isnull().sum()
    if unmapped:
        print(f"  Warning: {unmapped} unmapped labels - defaulting to R2L")
        df['category'] = df['category'].fillna('R2L')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df, _ = drop_constant_features(df, numeric_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = handle_nulls_and_inf(df, numeric_cols)

    le = LabelEncoder()
    y = le.fit_transform(df['category'])
    print(f"  Classes: {list(le.classes_)}")

    label_map = {i: c for i, c in enumerate(le.classes_)}
    np.save(SPLITS_DIR / "nslkdd_label_map.npy", label_map)

    X = df[numeric_cols].values.astype(np.float32)
    save_splits(X, y, "nslkdd")


### UNSW-NB15 ###

def process_unswnb15():
    print("\n=== UNSW-NB15 ===")

    df = pd.read_csv("data/UNSW-NB15/Data.csv", low_memory=False)
    labels = pd.read_csv("data/UNSW-NB15/Label.csv")
    df['Label'] = labels['Label'].values
    print(f"  Loaded: {df.shape}")

    # Map integer labels to attack category names
    label_map = {
        0: 'Normal',
        1: 'Fuzzers',
        2: 'Analysis',
        3: 'Backdoors',
        4: 'DoS',
        5: 'Exploits',
        6: 'Generic',
        7: 'Reconnaissance',
        8: 'Shellcode',
        9: 'Worms'
    }
    df['attack_cat'] = df['Label'].map(label_map)

    unmapped = df['attack_cat'].isnull().sum()
    if unmapped:
        print(f"  Warning: {unmapped} unmapped labels")
        print(df[df['attack_cat'].isnull()]['Label'].value_counts())

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {before - len(df)} duplicate rows")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Label']

    df, _ = drop_constant_features(df, numeric_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Label']
    df = handle_nulls_and_inf(df, numeric_cols)

    le = LabelEncoder()
    y = le.fit_transform(df['attack_cat'])
    print(f"  Classes: {list(le.classes_)}")

    label_map_encoded = {i: c for i, c in enumerate(le.classes_)}
    np.save(SPLITS_DIR / "unswnb15_label_map.npy", label_map_encoded)

    X = df[numeric_cols].values.astype(np.float32)
    save_splits(X, y, "unswnb15")


### Main ###

if __name__ == "__main__":
    process_cicids2017()
    process_nslkdd()
    process_unswnb15()
    print("\nAll datasets processed. Splits saved to data/splits/")
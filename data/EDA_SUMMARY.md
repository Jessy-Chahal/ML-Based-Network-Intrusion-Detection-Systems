# EDA Summary

EDA notebooks are in `notebooks/`. Processed splits are in `data/splits/`
(tracked via Git LFS). See `data/README.md` for raw data download instructions.

## CICIDS2017

| Property | Value |
|---|---|
| Source files | 8 CSVs (Monday–Friday traffic days) |
| Raw shape | 2,830,743 rows × 79 features |
| Shape after cleaning | ~2,522,362 rows × 71 features |
| Label column | `Label` (string) |
| Classes | 15 |

### Data Quality

**Null values:** None across all columns.

**Infinite values:** `Flow Bytes/s` has 1,509 infinite values and `Flow Packets/s` has 2,867. These come from zero-duration flows where CICFlowMeter divides by zero. Handled in `preprocess.py` by replacing with NaN and imputing the column mean.

**Negative values:** `Flow Duration` has a minimum of -13, which is physically impossible. This is a known CICFlowMeter artifact affecting very few rows and is retained as-is.

### Constant Features

The following 8 features are always 0 and are dropped in `preprocess.py`:
`Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`.

### Duplicates

308,381 rows (10.89%) are duplicates. They are dropped in `preprocess.py`.

### Class Distribution

| Label | Count | % |
|---|---|---|
| BENIGN | 2,273,097 | 80.30% |
| DoS Hulk | 231,073 | 8.16% |
| PortScan | 158,930 | 5.61% |
| DDoS | 128,027 | 4.52% |
| DoS GoldenEye | 10,293 | 0.36% |
| FTP-Patator | 7,938 | 0.28% |
| SSH-Patator | 5,897 | 0.21% |
| DoS slowloris | 5,796 | 0.20% |
| DoS Slowhttptest | 5,499 | 0.19% |
| Bot | 1,966 | 0.07% |
| Web Attack – Brute Force | 1,507 | 0.05% |
| Web Attack – XSS | 652 | 0.02% |
| Infiltration | 36 | <0.01% |
| Web Attack – Sql Injection | 21 | <0.01% |
| Heartbleed | 11 | <0.01% |

BENIGN traffic makes up 80.3% of the dataset. Heartbleed (11 samples), SQL Injection (21), and Infiltration (36) are severely underrepresented. Class-weighted loss should be used during model training to avoid the classifier simply learning to predict BENIGN for everything.

### Known Labeling Issues

The following classes have documented labeling errors in the original CICFlowMeter output and are retained as-is in the splits. Results for these classes should be interpreted carefully:

- **PortScan** — some flows are mislabeled
- **Bot** — some flows are mislabeled
- **DoS Hulk** — known CICFlowMeter bug affecting flow statistics


## NSL-KDD

| Property | Value |
|---|---|
| Source files | `KDDTrain+.txt`, `KDDTest+.txt` |
| Raw shape | 125,973 train + 22,544 test = 148,517 rows × 43 columns |
| Shape after cleaning | 148,517 rows × 40 features |
| Label column | `label` (string, mapped to 5-class taxonomy) |
| Classes | 5 (Normal, DoS, Probe, R2L, U2R) |

### Data Quality

**Null values:** None.

**Infinite values:** None.

**Duplicates:** None.

**Categorical features:** `protocol_type`, `service`, and `flag` are string columns that are label-encoded in `preprocess.py`.

**Non-feature column:** `difficulty` is present in the raw files but is not a network feature — it is dropped in `preprocess.py`.

### Constant Features

`num_outbound_cmds` is always 0 and is dropped in `preprocess.py`.

### Class Distribution

The 23 specific attack labels in the raw files are mapped to a 5-class taxonomy. Any labels present in the test set but not the training set default to R2L.

| Category | Count | % |
|---|---|---|
| Normal | 67,343 | 53.46% |
| DoS | 45,927 | 36.46% |
| Probe | 11,656 | 9.25% |
| R2L | 995 | 0.79% |
| U2R | 52 | 0.04% |

This dataset has the most severe class imbalance of the three. R2L and U2R together make up less than 1% of the training data, meaning a model that never predicts either class can still score above 99% accuracy. Per-class F1 scores should be reported rather than overall accuracy, and class-weighted loss is necessary for meaningful results.

## UNSW-NB15

| Property | Value |
|---|---|
| Source files | `Data.csv` (features), `Label.csv` (integer labels) |
| Raw shape | 447,915 rows × 76 features + 1 label column |
| Shape after cleaning | ~306,173 rows × 67 features |
| Label column | Integer 0–9 (joined from `Label.csv` on row index) |
| Classes | 10 |

### Data Quality

**Null values:** None.

**Infinite values:** None. This is the cleanest of the three datasets and required no imputation.

### Constant Features

The following 9 features are always 0 and are dropped in `preprocess.py`:
`Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`, `URG Flag Count`, `CWR Flag Count`, `ECE Flag Count`, `Fwd Bytes/Bulk Avg`, `Fwd Packet/Bulk Avg`, `Fwd Bulk Rate Avg`.

### Duplicates

141,742 rows (31.64%) are duplicates. These are dropped in `preprocess.py`.

### Label Structure

Labels are stored as integers in a separate `Label.csv` and joined to `Data.csv` on row index. The mapping is:

| Integer | Category |
|---|---|
| 0 | Normal |
| 1 | Fuzzers |
| 2 | Analysis |
| 3 | Backdoors |
| 4 | DoS |
| 5 | Exploits |
| 6 | Generic |
| 7 | Reconnaissance |
| 8 | Shellcode |
| 9 | Worms |

All 10 labels mapped successfully with no unmapped values.

### Class Distribution

| Category | Count | % |
|---|---|---|
| Normal | 358,332 | 80.00% |
| DoS | 30,951 | 6.91% |
| Exploits | 29,613 | 6.61% |
| Reconnaissance | 16,735 | 3.74% |
| Generic | 4,632 | 1.03% |
| Backdoors | 4,467 | 1.00% |
| Shellcode | 2,102 | 0.47% |
| Analysis | 452 | 0.10% |
| Fuzzers | 385 | 0.09% |
| Worms | 246 | 0.05% |

Normal traffic is 80% of the dataset. Worms, Fuzzers, and Analysis are very sparse after deduplication and will need class-weighted loss to train meaningful classifiers.
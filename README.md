# ML-Based Network Intrusion Detection Systems

This project studies how adversarial attacks can fool ML-based Network Intrusion Detection Systems (NIDS), and how well adversarial retraining can recover robustness. The attacks are constrained to be realistic - they must remain valid at the network level, still accomplish their original goal, and be achievable without direct model access.

---

## Overview

We train classifiers on the CICDS2017 datases, then generate adversarial examples using three families of evasion attacks. Each adversarial example must pass four constraint checks before it's used. Finally, we retrain the models on adversarial data and measure how much robustness is recovered.


**Models**

Each dataset gets three individually trained classifiers that are combined into a majority-voting ensemble:

- Random Forest (RF)
- XGBoost
- Multi-Layer Perceptron (MLP)

---

## Attack Categories

### Category A - Feature Obfuscation

Dilutes or masks anomalous flow statistics by injecting decoy packets, blending with benign traffic, or slowing port scan probes. The goal is to push the classifier's input closer to the benign distribution without changing the underlying attack behavior.

### Category B - Behavioral Mimicry

Adjusts the timing (inter-arrival times) and packet sizes of malicious flows to match the statistical profile of normal application traffic (e.g., HTTPS keep-alive). The attack traffic looks like background traffic at the feature level.

### Category C - Protocol Exploitation

Manipulates TCP-level features through payload fragmentation, injecting benign-looking TCP options (timestamps, window scaling), and shifting ACK timing. Each mutation corresponds to a real network action - no abstract arithmetic on the feature vector.

---

## Constraints

Every adversarial example must pass four checks before it's added to any dataset or used in evaluation. Samples that fail are discarded.

| Constraint | What it enforces |
|---|---|
| **Protocol Validity** | The flow must be producible by a real TCP/IP stack (valid flag combos, consistent header/packet/byte counts, non-negative duration) |
| **Functional Preservation** | The attack must still work - a DoS flow can't be slowed below the threshold needed to cause disruption |
| **Behavioral Plausibility** | The flow must not trigger simple rule-based detectors (packet rates within physical limits, realistic packet sizes, plausible IAT) |
| **Attacker Capability (Gray-Box)** | No white-box gradients, no model queries - mutations must correspond to real network operations |

Constraints are enforced in `src/constraints.py`. Thresholds are documented in `docs/constraint_thresholds.md`.

---

## Defense

The defense is adversarial retraining: augment the training set with constraint-validated adversarial examples, then retrain from scratch. We evaluate three variants:

- **Full** - trained on all three attack families combined
- **Partial A / B / C** - trained on one attack family only (to measure cross-family generalization)

Evaluation compares baseline vs. adversarially retrained ensemble on clean test data and on held-out adversarial examples.

---

## Project Structure

```
src/
  attacks/
    feature_obfuscation.py   # Category A
    behavioral_mimicry.py    # Category B
    protocol_exploitation.py # Category C
    evaluate_attack_{a,b,c}.py
  defense/
    ensemble.py              # RF + XGBoost + MLP majority-vote ensemble
    gen_adversarial_dataset.py
    gen_adversarial_partial.py
    train_adversarial.py
    evaluate_defense.py
  model/
    train_baseline.py
    preprocess.py
    evaluate_adv_training_clean.py
  constraints.py             # Constraint validators
  mutations.py               # Atomic network-level mutations
  dotenv_utils.py

data/
  splits/                    # Preprocessed .npz splits (Git LFS)
  adversarial/               # Generated adversarial splits
  README.md                  # Dataset download instructions

models/                      # Trained model files (.pkl, .h5)
results/                     # Metrics JSON files
notebooks/                   # EDA notebooks (CICIDS2017, NSL-KDD, UNSW-NB15)
docs/                        # Constraint specs, feature notes
tests/                       # pytest tests
```

---
## Running the Project

The project can be most easily run through the docker image found [here](https://drive.google.com/drive/folders/1KnOcGRnqUJwCtyPUR4IYc0v1WrXPPrZn?usp=sharing) (instructions included in zip).

Otherwise, the steps below may be followed instead.

---

## Setup

This project uses Conda.

```bash
conda env create -f environment.yml
conda activate nids-adv
```

**Key dependencies:** Python 3.13, TensorFlow 2.21, Keras 3.13, scikit-learn 1.8, XGBoost 2.x

---

## Data

The raw datasets are not included in the repo. Download each one and place files as described in [data/README.md](data/README.md). The preprocessed splits in `data/splits/` are tracked via Git LFS and are ready to use once pulled.

---

## Running the Pipeline

**Train baseline models (all datasets):**
```bash
python -m src.model.train_baseline
```

**Generate adversarial examples:**
```bash
python -m src.defense.gen_adversarial_dataset
```

**Evaluate an attack family:**
```bash
python src/attacks/evaluate_attack_a.py
python src/attacks/evaluate_attack_b.py
python src/attacks/evaluate_attack_c.py
```

**Retrain on adversarial data:**
```bash
# Full adversarial training (all families)
python -m src.model.train_adversarial --all

# Partial - single attack family
python -m src.model.train_adversarial --all --attack a
```

**Evaluate defense:**
```bash
python src/defense/evaluate_defense.py
python src/defense/evaluate_defense.py --attack a
```


#!/usr/bin/env bash
set -euo pipefail

# Runs the project pipeline in the exact requested order.
# Usage:
#   bash scripts/run_full_pipeline.sh
# Optional:
#   PYTHON_BIN=python3 bash scripts/run_full_pipeline.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
TOTAL_STEPS=21

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_step() {
  local step="$1"
  shift
  echo
  echo "[$step/$TOTAL_STEPS] $*"
  "$@"
}

echo "Running full pipeline from: $ROOT_DIR"
echo "Python executable: $PYTHON_BIN"

# 1. train_baseline.py
run_step 1 "$PYTHON_BIN" src/model/train_baseline.py --all

# 2. evaluate_attack_a.py
run_step 2 "$PYTHON_BIN" src/attacks/evaluate_attack_a.py

# 3. evaluate_attack_b.py
run_step 3 "$PYTHON_BIN" src/attacks/evaluate_attack_b.py

# 4. evaluate_attack_c.py
run_step 4 "$PYTHON_BIN" src/attacks/evaluate_attack_c.py

# 5. gen_adversarial_dataset.py
run_step 5 "$PYTHON_BIN" src/model/gen_adversarial_dataset.py

# 6. gen_adversarial_partial.py --attack a
run_step 6 "$PYTHON_BIN" src/model/gen_adversarial_partial.py --attack a

# 7. gen_adversarial_partial.py --attack b
run_step 7 "$PYTHON_BIN" src/model/gen_adversarial_partial.py --attack b

# 8. gen_adversarial_partial.py --attack c
run_step 8 "$PYTHON_BIN" src/model/gen_adversarial_partial.py --attack c

# 9. train_adversarial.py --all
run_step 9 "$PYTHON_BIN" src/model/train_adversarial.py --all

# 10. train_adversarial.py --all --attack a
run_step 10 "$PYTHON_BIN" src/model/train_adversarial.py --all --attack a

# 11. train_adversarial.py --all --attack b
run_step 11 "$PYTHON_BIN" src/model/train_adversarial.py --all --attack b

# 12. train_adversarial.py --all --attack c
run_step 12 "$PYTHON_BIN" src/model/train_adversarial.py --all --attack c

# 13. evaluate_adv_training_clean.py
run_step 13 "$PYTHON_BIN" src/evaluate_adv_training_clean.py

# 14. evaluate_defense.py
run_step 14 "$PYTHON_BIN" src/defense/evaluate_defense.py

# 15. evaluate_defense.py --attack a
run_step 15 "$PYTHON_BIN" src/defense/evaluate_defense.py --attack a

# 16. evaluate_defense.py --attack b
run_step 16 "$PYTHON_BIN" src/defense/evaluate_defense.py --attack b

# 17. evaluate_defense.py --attack c
run_step 17 "$PYTHON_BIN" src/defense/evaluate_defense.py --attack c

# 18. summarize_team_metrics.py
run_step 18 "$PYTHON_BIN" scripts/summarize_team_metrics.py

# 19. summarize_team_metrics.py --attack a
run_step 19 "$PYTHON_BIN" scripts/summarize_team_metrics.py --attack a

# 20. summarize_team_metrics.py --attack b
run_step 20 "$PYTHON_BIN" scripts/summarize_team_metrics.py --attack b

# 21. summarize_team_metrics.py --attack c
run_step 21 "$PYTHON_BIN" scripts/summarize_team_metrics.py --attack c

echo
echo "Pipeline completed successfully."

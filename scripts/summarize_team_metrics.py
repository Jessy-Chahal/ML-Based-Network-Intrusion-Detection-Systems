"""
Summarize team attack/defense metrics by owner and team average.

Scans owner-prefixed JSON files in results/ (shad/alyssa/jessy by default),
prints a terminal summary, and writes a detailed JSON report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_OWNERS = ["shad", "alyssa", "jessy"]
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "team_metrics_summary.json"
DEFAULT_MAX_WARNINGS = 20


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(mean(vals))


def _safe_median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(median(vals))


def _round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _group_mean(rows: List[Dict[str, Any]], key_name: str, value_name: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        key = row.get(key_name)
        val = _as_float(row.get(value_name))
        if key is None or val is None:
            continue
        grouped[str(key)].append(val)
    return {k: round(float(mean(vs)), 6) for k, vs in sorted(grouped.items())}


def _metric_threshold(metric_key: str) -> float:
    if "n_mutations_meeting_target" in metric_key:
        return 2.0
    if "n_mutations_evaluated" in metric_key:
        return 2.0
    if "dataset.overall_pass" in metric_key:
        return 0.75
    if "recovery_delta_pp" in metric_key:
        return 15.0
    if "accuracy_drop_pp" in metric_key:
        return 2.0
    if "detection_rate" in metric_key:
        return 0.10
    if "accuracy" in metric_key:
        return 0.05
    if ".esr" in metric_key:
        return 0.20
    if "constraint_satisfaction_rate" in metric_key:
        return 0.20
    if "success_rate" in metric_key:
        return 0.20
    return 0.15


def _compute_skew_warnings(team_metrics: Dict[str, Dict[str, Any]], owners: List[str]) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    owner_set = set(owners)

    for metric_key, detail in team_metrics.items():
        values_by_owner = detail.get("values_by_owner", {})
        if set(values_by_owner.keys()) != owner_set:
            continue

        vals = {o: float(values_by_owner[o]) for o in owners}
        med = float(median(vals.values()))
        threshold = _metric_threshold(metric_key)

        deviations = {o: abs(v - med) for o, v in vals.items()}
        worst_owner = max(deviations, key=deviations.get)
        worst_dev = deviations[worst_owner]

        if worst_dev >= threshold:
            severity = "high" if worst_dev >= 2.0 * threshold else "medium"
            warnings.append(
                {
                    "severity": severity,
                    "metric": metric_key,
                    "owner_with_max_deviation": worst_owner,
                    "owner_value": round(vals[worst_owner], 6),
                    "team_median": round(med, 6),
                    "absolute_deviation": round(worst_dev, 6),
                    "threshold": threshold,
                    "values_by_owner": {k: round(v, 6) for k, v in vals.items()},
                    "note": "One owner differs substantially from the team median for this metric.",
                }
            )

    warnings.sort(
        key=lambda w: (
            0 if w["severity"] == "high" else 1,
            -w["absolute_deviation"],
            w["metric"],
        )
    )
    return warnings


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_attack(
    owner: str,
    attack_name: str,
    payload: Dict[str, Any],
    metric_store: Dict[str, float],
) -> Dict[str, Any]:
    esr_rows: List[Dict[str, Any]] = []
    csr_rows: List[Dict[str, Any]] = []
    success_rows: List[Dict[str, Any]] = []
    compatibility_notes: List[str] = []
    zero_success_mutations = 0

    for dataset_blob in payload.get("results", []):
        dataset = str(dataset_blob.get("dataset", "unknown"))
        mutation_metrics = dataset_blob.get("metrics", {})

        for mutation, m_blob in mutation_metrics.items():
            mutation = str(mutation)

            n_samples = _as_float(m_blob.get("n_samples"))
            n_success = _as_float(m_blob.get("n_mutation_success"))
            if n_samples is not None and n_samples > 0 and n_success is not None:
                success_rate = n_success / n_samples
                success_rows.append(
                    {
                        "dataset": dataset,
                        "mutation": mutation,
                        "success_rate": success_rate,
                    }
                )
                metric_store[
                    f"{attack_name}.success_rate.{dataset}.{mutation}"
                ] = float(success_rate)
                if n_success == 0:
                    zero_success_mutations += 1

            csr = _as_float(m_blob.get("constraint_satisfaction_rate"))
            if csr is not None:
                csr_rows.append({"dataset": dataset, "mutation": mutation, "csr": csr})
                metric_store[
                    f"{attack_name}.constraint_satisfaction_rate.{dataset}.{mutation}"
                ] = float(csr)

            note = m_blob.get("compatibility_note")
            if isinstance(note, str) and note.strip():
                compatibility_notes.append(note.strip())

            models = m_blob.get("models", {})
            for model_name, model_blob in models.items():
                model = str(model_name)
                esr = _as_float(model_blob.get("esr"))
                if esr is not None:
                    esr_rows.append(
                        {
                            "dataset": dataset,
                            "mutation": mutation,
                            "model": model,
                            "esr": esr,
                        }
                    )
                    metric_store[
                        f"{attack_name}.esr.{dataset}.{mutation}.{model}"
                    ] = float(esr)

                esr_on_valid = _as_float(model_blob.get("esr_on_valid"))
                if esr_on_valid is not None:
                    metric_store[
                        f"{attack_name}.esr_on_valid.{dataset}.{mutation}.{model}"
                    ] = float(esr_on_valid)

    summary = {
        "owner": owner,
        "attack": attack_name,
        "n_esr_points": len(esr_rows),
        "overall_mean_esr": _round_or_none(_safe_mean(r["esr"] for r in esr_rows)),
        "dataset_mean_esr": _group_mean(esr_rows, "dataset", "esr"),
        "model_mean_esr": _group_mean(esr_rows, "model", "esr"),
        "mutation_mean_esr": _group_mean(esr_rows, "mutation", "esr"),
        "overall_mean_constraint_satisfaction_rate": _round_or_none(
            _safe_mean(r["csr"] for r in csr_rows)
        ),
        "overall_mean_success_rate": _round_or_none(
            _safe_mean(r["success_rate"] for r in success_rows)
        ),
        "n_zero_success_mutation_entries": zero_success_mutations,
        "compatibility_notes_count": len(compatibility_notes),
        "compatibility_notes": sorted(set(compatibility_notes)),
    }
    return summary


def _extract_defense_metrics(
    payload: Dict[str, Any],
    metric_store: Dict[str, float],
) -> Dict[str, Any]:
    def _extract_one_dataset(ds_blob: Dict[str, Any]) -> Dict[str, Any]:
        ds = str(ds_blob.get("dataset", "unknown"))
        clean = ds_blob.get("clean_detection", {})

        baseline_clean = _as_float(clean.get("baseline_detection_rate"))
        adv_clean = _as_float(clean.get("adversarial_detection_rate"))
        if baseline_clean is not None:
            metric_store[f"defense.clean.baseline_detection_rate.{ds}"] = baseline_clean
        if adv_clean is not None:
            metric_store[f"defense.clean.adversarial_detection_rate.{ds}"] = adv_clean

        n_eval = _as_float(ds_blob.get("n_mutations_evaluated"))
        n_hit = _as_float(ds_blob.get("n_mutations_meeting_target"))
        if n_eval is not None:
            metric_store[f"defense.n_mutations_evaluated.{ds}"] = n_eval
        if n_hit is not None:
            metric_store[f"defense.n_mutations_meeting_target.{ds}"] = n_hit
        if n_eval and n_hit is not None and n_eval > 0:
            metric_store[f"defense.target_hit_rate.{ds}"] = n_hit / n_eval

        mutation_rows: List[Dict[str, Any]] = []
        for m in ds_blob.get("mutations", []):
            if m.get("skipped"):
                continue
            mutation = str(m.get("mutation", "unknown"))
            base_dr = _as_float(m.get("baseline_detection_rate"))
            adv_dr = _as_float(m.get("adversarial_detection_rate"))
            delta_pp = _as_float(m.get("recovery_delta_pp"))

            if base_dr is not None:
                metric_store[f"defense.mutation.baseline_detection_rate.{ds}.{mutation}"] = base_dr
            if adv_dr is not None:
                metric_store[f"defense.mutation.adversarial_detection_rate.{ds}.{mutation}"] = adv_dr
            if delta_pp is not None:
                metric_store[f"defense.mutation.recovery_delta_pp.{ds}.{mutation}"] = delta_pp
                mutation_rows.append({"mutation": mutation, "recovery_delta_pp": delta_pp})

            # Per-model detection rates — baseline
            for model, dr in m.get("baseline_per_model", {}).items():
                val = _as_float(dr)
                if val is not None:
                    metric_store[
                        f"defense.mutation.baseline_detection_rate_per_model.{ds}.{mutation}.{model}"
                    ] = val

            # Per-model detection rates — adversarial
            for model, dr in m.get("adversarial_per_model", {}).items():
                val = _as_float(dr)
                if val is not None:
                    metric_store[
                        f"defense.mutation.adversarial_detection_rate_per_model.{ds}.{mutation}.{model}"
                    ] = val

            # Per-model recovery delta
            base_per = m.get("baseline_per_model", {})
            adv_per  = m.get("adversarial_per_model", {})
            for model in set(base_per) & set(adv_per):
                b = _as_float(base_per[model])
                a = _as_float(adv_per[model])
                if b is not None and a is not None:
                    metric_store[
                        f"defense.mutation.recovery_delta_pp_per_model.{ds}.{mutation}.{model}"
                    ] = round((a - b) * 100.0, 4)

        best = None
        worst = None
        if mutation_rows:
            best = max(mutation_rows, key=lambda r: r["recovery_delta_pp"])
            worst = min(mutation_rows, key=lambda r: r["recovery_delta_pp"])

        target_hit_rate = (
            (n_hit / n_eval) if (n_eval and n_hit is not None and n_eval > 0) else None
        )
        return {
            "dataset": ds,
            "target_recovery_pp": ds_blob.get("target_recovery_pp"),
            "n_attack_samples": ds_blob.get("n_attack_samples"),
            "n_mutations_evaluated": int(n_eval) if n_eval is not None else None,
            "n_mutations_skipped": (
                int(ds_blob["n_mutations_skipped"])
                if _as_float(ds_blob.get("n_mutations_skipped")) is not None
                else None
            ),
            "n_mutations_meeting_target": int(n_hit) if n_hit is not None else None,
            "target_hit_rate": _round_or_none(target_hit_rate),
            "clean_baseline_detection_rate": _round_or_none(baseline_clean),
            "clean_adversarial_detection_rate": _round_or_none(adv_clean),
            "mean_recovery_delta_pp": _round_or_none(
                _safe_mean(r["recovery_delta_pp"] for r in mutation_rows), digits=4
            ),
            "median_recovery_delta_pp": _round_or_none(
                _safe_median(r["recovery_delta_pp"] for r in mutation_rows), digits=4
            ),
            "best_mutation_by_recovery": best,
            "worst_mutation_by_recovery": worst,
        }

    # New schema: {"datasets": [ ... ]}
    dataset_entries = payload.get("datasets")
    per_dataset_list: List[Dict[str, Any]] = []
    if isinstance(dataset_entries, list) and dataset_entries:
        for ds_blob in dataset_entries:
            if isinstance(ds_blob, dict):
                per_dataset_list.append(_extract_one_dataset(ds_blob))
    else:
        # Backward compatibility with old single-dataset schema
        if isinstance(payload, dict):
            per_dataset_list.append(_extract_one_dataset(payload))

    per_dataset: Dict[str, Dict[str, Any]] = {
        str(d.get("dataset", "unknown")): d for d in per_dataset_list
    }

    eval_totals = [
        int(d["n_mutations_evaluated"])
        for d in per_dataset_list
        if isinstance(d.get("n_mutations_evaluated"), int)
    ]
    hit_totals = [
        int(d["n_mutations_meeting_target"])
        for d in per_dataset_list
        if isinstance(d.get("n_mutations_meeting_target"), int)
    ]
    clean_base_vals = [
        float(d["clean_baseline_detection_rate"])
        for d in per_dataset_list
        if _as_float(d.get("clean_baseline_detection_rate")) is not None
    ]
    clean_adv_vals = [
        float(d["clean_adversarial_detection_rate"])
        for d in per_dataset_list
        if _as_float(d.get("clean_adversarial_detection_rate")) is not None
    ]
    mean_delta_vals = [
        float(d["mean_recovery_delta_pp"])
        for d in per_dataset_list
        if _as_float(d.get("mean_recovery_delta_pp")) is not None
    ]

    total_eval = sum(eval_totals) if eval_totals else None
    total_hit = sum(hit_totals) if hit_totals else None
    overall_hit_rate = (
        (total_hit / total_eval)
        if (total_eval is not None and total_eval > 0 and total_hit is not None)
        else None
    )

    if total_eval is not None:
        metric_store["defense.n_mutations_evaluated.overall"] = float(total_eval)
    if total_hit is not None:
        metric_store["defense.n_mutations_meeting_target.overall"] = float(total_hit)
    if overall_hit_rate is not None:
        metric_store["defense.target_hit_rate.overall"] = float(overall_hit_rate)
    if clean_base_vals:
        metric_store["defense.clean.baseline_detection_rate.overall"] = float(mean(clean_base_vals))
    if clean_adv_vals:
        metric_store["defense.clean.adversarial_detection_rate.overall"] = float(mean(clean_adv_vals))
    if mean_delta_vals:
        metric_store["defense.mean_recovery_delta_pp.overall"] = float(mean(mean_delta_vals))

    return {
        "n_datasets": len(per_dataset_list),
        "per_dataset": per_dataset,
        "overall": {
            "n_mutations_evaluated_total": total_eval,
            "n_mutations_meeting_target_total": total_hit,
            "target_hit_rate_overall": _round_or_none(overall_hit_rate),
            "mean_clean_baseline_detection_rate": _round_or_none(_safe_mean(clean_base_vals)),
            "mean_clean_adversarial_detection_rate": _round_or_none(_safe_mean(clean_adv_vals)),
            "mean_recovery_delta_pp": _round_or_none(_safe_mean(mean_delta_vals), digits=4),
            "median_recovery_delta_pp": _round_or_none(_safe_median(mean_delta_vals), digits=4),
        },
    }


def _extract_adv_training_clean(
    payload: Dict[str, Any],
    metric_store: Dict[str, float],
) -> Dict[str, Any]:
    dataset_rows: List[Dict[str, Any]] = []
    drop_rows: List[Dict[str, Any]] = []
    missing_baseline_models = 0
    violations = 0

    datasets = payload.get("datasets", {})
    for dataset, d_blob in datasets.items():
        ds = str(dataset)
        overall_pass = d_blob.get("overall_pass")
        if isinstance(overall_pass, bool):
            metric_store[f"adv_training_clean.dataset.overall_pass.{ds}"] = 1.0 if overall_pass else 0.0

        models = d_blob.get("models", {})
        for model, m_blob in models.items():
            model = str(model)
            base_acc = _as_float(m_blob.get("baseline_accuracy"))
            retr_acc = _as_float(m_blob.get("retrained_clean_test_accuracy"))
            drop_pp = _as_float(m_blob.get("accuracy_drop_pp"))
            within = m_blob.get("within_3pp_target")

            if base_acc is not None:
                metric_store[f"adv_training_clean.baseline_accuracy.{ds}.{model}"] = base_acc
            else:
                missing_baseline_models += 1
            if retr_acc is not None:
                metric_store[f"adv_training_clean.retrained_accuracy.{ds}.{model}"] = retr_acc
            if drop_pp is not None:
                metric_store[f"adv_training_clean.accuracy_drop_pp.{ds}.{model}"] = drop_pp
                drop_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "accuracy_drop_pp": drop_pp,
                    }
                )
            if isinstance(within, bool) and (not within):
                violations += 1

            if base_acc is not None and retr_acc is not None:
                dataset_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "baseline_accuracy": base_acc,
                        "retrained_accuracy": retr_acc,
                    }
                )

    worst_drop = None
    if drop_rows:
        worst_drop = max(drop_rows, key=lambda r: r["accuracy_drop_pp"])

    return {
        "overall_pass": payload.get("overall_pass"),
        "mean_accuracy_drop_pp": _round_or_none(
            _safe_mean(r["accuracy_drop_pp"] for r in drop_rows), digits=4
        ),
        "median_accuracy_drop_pp": _round_or_none(
            _safe_median(r["accuracy_drop_pp"] for r in drop_rows), digits=4
        ),
        "worst_accuracy_drop_entry": worst_drop,
        "n_within_target_violations": violations,
        "n_missing_baseline_model_entries": missing_baseline_models,
        "dataset_mean_retrained_accuracy": _group_mean(
            dataset_rows, "dataset", "retrained_accuracy"
        ),
    }


def _extract_retrained_adv(
    payload: Dict[str, Any],
    metric_store: Dict[str, float],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for dataset, d_blob in payload.items():
        if not isinstance(d_blob, dict):
            continue
        ds = str(dataset)
        for model, m_blob in d_blob.items():
            if not isinstance(m_blob, dict):
                continue
            model = str(model)
            acc = _as_float(m_blob.get("accuracy"))
            if acc is None:
                continue
            rows.append({"dataset": ds, "model": model, "accuracy": acc})
            metric_store[f"retrained_adversarial.accuracy.{ds}.{model}"] = acc

    return {
        "n_accuracy_entries": len(rows),
        "overall_mean_accuracy": _round_or_none(_safe_mean(r["accuracy"] for r in rows)),
        "dataset_mean_accuracy": _group_mean(rows, "dataset", "accuracy"),
        "model_mean_accuracy": _group_mean(rows, "model", "accuracy"),
    }


def _resolve_owner_files(results_dir: Path, owner: str) -> Dict[str, Path]:
    files: Dict[str, Path] = {}

    files[f"{owner}_attack_a"] = results_dir / f"{owner}_attack_a_metrics_all_datasets.json"
    files[f"{owner}_attack_b"] = results_dir / f"{owner}_attack_b_metrics_all_datasets.json"
    files[f"{owner}_attack_c"] = results_dir / f"{owner}_attack_c_metrics_all_datasets.json"
    files[f"{owner}_defense"] = results_dir / f"{owner}_defense_metrics.json"
    files[f"{owner}_adv_training_clean"] = results_dir / f"{owner}_adv_training_clean_metrics.json"
    files[f"{owner}_retrained_adversarial"] = results_dir / f"{owner}_retrained_adversarial_metrics.json"
    return files


def _print_terminal_summary(
    owners: List[str],
    owner_summaries: Dict[str, Dict[str, Any]],
    team_metrics: Dict[str, Dict[str, Any]],
    skew_warnings: List[Dict[str, Any]],
    max_warnings: int,
):
    print("\n=== Team Metrics Summary ===")
    print(f"Owners: {', '.join(owners)}")
    print(f"Comparable metric keys (all owners present): {len(team_metrics)}")
    print(f"Skew warnings: {len(skew_warnings)}")

    print("\n--- Per-person summary ---")
    for owner in owners:
        blob = owner_summaries.get(owner, {})
        attacks = blob.get("attacks", {})
        defense = blob.get("defense", {})
        defense_overall = defense.get("overall", {})
        defense_by_ds = defense.get("per_dataset", {})
        adv_clean = blob.get("adv_training_clean", {})
        retr = blob.get("retrained_adversarial", {})

        a_esr = attacks.get("attack_a", {}).get("overall_mean_esr")
        b_esr = attacks.get("attack_b", {}).get("overall_mean_esr")
        c_esr = attacks.get("attack_c", {}).get("overall_mean_esr")
        hit = defense_overall.get("n_mutations_meeting_target_total")
        total = defense_overall.get("n_mutations_evaluated_total")
        mean_recovery = defense_overall.get("mean_recovery_delta_pp")

        print(f"\n{owner.upper()}:")
        print(f"  Attack mean ESR (A/B/C): {a_esr} / {b_esr} / {c_esr}")
        print(f"  Defense target hits (total): {hit}/{total}; mean recovery_delta_pp: {mean_recovery}")
        print(
            "  Clean detection mean (baseline/adversarial): "
            f"{defense_overall.get('mean_clean_baseline_detection_rate')} / "
            f"{defense_overall.get('mean_clean_adversarial_detection_rate')}"
        )
        if defense_by_ds:
            ds_bits = []
            for ds_name in sorted(defense_by_ds):
                ds_obj = defense_by_ds[ds_name]
                ds_bits.append(
                    f"{ds_name}: {ds_obj.get('n_mutations_meeting_target')}/"
                    f"{ds_obj.get('n_mutations_evaluated')}"
                )
            print(f"  Defense per-dataset target hits: {' | '.join(ds_bits)}")
        print(
            "  Adv-train clean mean drop (pp): "
            f"{adv_clean.get('mean_accuracy_drop_pp')} | violations: "
            f"{adv_clean.get('n_within_target_violations')}"
        )
        print(
            "  Retrained adversarial mean accuracy: "
            f"{retr.get('overall_mean_accuracy')}"
        )

    print("\n--- Team average snapshots ---")
    snapshot_keys = [
        "defense.clean.baseline_detection_rate.cicids2017",
        "defense.clean.adversarial_detection_rate.cicids2017",
        "defense.target_hit_rate.cicids2017",
        "defense.target_hit_rate.overall",
        "attack_a.esr.cicids2017.inject_decoy_flows.random_forest",
        "attack_b.esr.cicids2017.mimic_packet_size.mlp",
        "attack_c.esr.cicids2017.fragment_payload.mlp",
    ]
    for key in snapshot_keys:
        row = team_metrics.get(key)
        if row is None:
            continue
        print(f"  {key}: avg={row['average']}, values={row['values_by_owner']}")

    if skew_warnings:
        print("\n--- Top skew warnings ---")
        for w in skew_warnings[:max_warnings]:
            print(
                f"  [{w['severity']}] {w['metric']} | "
                f"owner={w['owner_with_max_deviation']} "
                f"value={w['owner_value']} median={w['team_median']} "
                f"deviation={w['absolute_deviation']} threshold={w['threshold']}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize owner-prefixed attack/defense result JSON files and "
            "compute team averages with skew warnings."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing metric JSON files.",
    )
    parser.add_argument(
        "--owners",
        nargs="+",
        default=DEFAULT_OWNERS,
        help="Owner prefixes to include (default: shad alyssa jessy).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path for summary report.",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=DEFAULT_MAX_WARNINGS,
        help="Max skew warnings to print to terminal.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    owners = [str(o).strip().lower() for o in args.owners if str(o).strip()]

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    owner_metric_store: Dict[str, Dict[str, float]] = {}
    owner_summaries: Dict[str, Dict[str, Any]] = {}
    files_by_owner: Dict[str, Dict[str, Any]] = {}

    for owner in owners:
        owner_metrics: Dict[str, float] = {}
        owner_summary: Dict[str, Any] = {
            "attacks": {},
            "defense": {},
            "adv_training_clean": {},
            "retrained_adversarial": {},
            "missing_files": [],
        }

        paths = _resolve_owner_files(results_dir, owner)
        file_report: Dict[str, Any] = {}

        for file_key, path in paths.items():
            exists = path.exists()
            file_report[file_key] = {"path": str(path), "exists": exists}
            if not exists:
                owner_summary["missing_files"].append(str(path))
                continue

            payload = _load_json(path)
            if file_key.endswith("_attack_a"):
                owner_summary["attacks"]["attack_a"] = _extract_attack(
                    owner=owner,
                    attack_name="attack_a",
                    payload=payload,
                    metric_store=owner_metrics,
                )
            elif file_key.endswith("_attack_b"):
                owner_summary["attacks"]["attack_b"] = _extract_attack(
                    owner=owner,
                    attack_name="attack_b",
                    payload=payload,
                    metric_store=owner_metrics,
                )
            elif file_key.endswith("_attack_c"):
                owner_summary["attacks"]["attack_c"] = _extract_attack(
                    owner=owner,
                    attack_name="attack_c",
                    payload=payload,
                    metric_store=owner_metrics,
                )
            elif file_key.endswith("_defense"):
                owner_summary["defense"] = _extract_defense_metrics(
                    payload=payload,
                    metric_store=owner_metrics,
                )
            elif file_key.endswith("_adv_training_clean"):
                owner_summary["adv_training_clean"] = _extract_adv_training_clean(
                    payload=payload,
                    metric_store=owner_metrics,
                )
            elif file_key.endswith("_retrained_adversarial"):
                owner_summary["retrained_adversarial"] = _extract_retrained_adv(
                    payload=payload,
                    metric_store=owner_metrics,
                )

        owner_metric_store[owner] = owner_metrics
        owner_summaries[owner] = owner_summary
        files_by_owner[owner] = file_report

    combined_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
    for owner, metric_map in owner_metric_store.items():
        for metric_key, value in metric_map.items():
            combined_metrics[metric_key][owner] = value

    team_metrics: Dict[str, Dict[str, Any]] = {}
    missing_owner_metrics: Dict[str, Dict[str, Any]] = {}
    for key in sorted(combined_metrics):
        values_by_owner = combined_metrics[key]
        avg_val = _safe_mean(values_by_owner.values())
        row = {
            "average": _round_or_none(avg_val),
            "values_by_owner": {
                owner: _round_or_none(float(values_by_owner[owner]))
                for owner in sorted(values_by_owner)
            },
            "n_owners_present": len(values_by_owner),
        }
        if len(values_by_owner) == len(owners):
            team_metrics[key] = row
        else:
            missing_owner_metrics[key] = row

    skew_warnings = _compute_skew_warnings(team_metrics, owners)
    warning_counts = Counter(w["owner_with_max_deviation"] for w in skew_warnings)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "owners": owners,
        "files_by_owner": files_by_owner,
        "per_person_summary": owner_summaries,
        "team_average": {
            "n_metric_keys_all_owners": len(team_metrics),
            "n_metric_keys_missing_owners": len(missing_owner_metrics),
            "metrics_all_owners": team_metrics,
            "metrics_missing_owners": missing_owner_metrics,
        },
        "skew_analysis": {
            "n_warnings": len(skew_warnings),
            "warning_counts_by_owner": dict(sorted(warning_counts.items())),
            "warnings": skew_warnings,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    _print_terminal_summary(
        owners=owners,
        owner_summaries=owner_summaries,
        team_metrics=team_metrics,
        skew_warnings=skew_warnings,
        max_warnings=max(0, int(args.max_warnings)),
    )
    print(f"\nSaved JSON summary: {output_path}")


if __name__ == "__main__":
    main()

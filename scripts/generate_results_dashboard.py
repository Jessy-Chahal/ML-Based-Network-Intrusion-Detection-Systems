import json
import os

def generate_static_dashboard(json_path="results/team_metrics_summary.json", output_path="results/results_summary_dashboard.html"):
    # 1. Pull data from the JSON file
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}. Please ensure it is in the same directory.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    owners = data.get("owners", [])
    per_person = data.get("per_person_summary", {})
    generated_at = data.get("generated_at_utc", "Unknown")

    # 2. Extract Metrics for Visualization
    formatted_owners = [o.capitalize() for o in owners]

    # Attack Metrics
    esr_a = [per_person[o]["attacks"]["attack_a"]["overall_mean_esr"] for o in owners]
    esr_b = [per_person[o]["attacks"]["attack_b"]["overall_mean_esr"] for o in owners]
    esr_c = [per_person[o]["attacks"]["attack_c"]["overall_mean_esr"] for o in owners]

    # Model Retrained Accuracy
    models = ["mlp", "random_forest", "xgboost", "lstm", "majority_voting"]
    model_acc = {m: [per_person[o]["retrained_adversarial"]["model_mean_accuracy"][m] for o in owners] for m in models}

    # Deep Dive Defense Metrics
    # Overall Recovery & Hit Rate
    recovery_overall = [per_person[o]["defense"]["overall"]["mean_recovery_delta_pp"] for o in owners]
    hit_rate_overall = [per_person[o]["defense"]["overall"]["target_hit_rate_overall"] * 100 for o in owners] # As percentage

    # Recovery Delta by Dataset
    rec_cicids = [per_person[o]["defense"]["per_dataset"]["cicids2017"]["mean_recovery_delta_pp"] for o in owners]
    rec_nslkdd = [per_person[o]["defense"]["per_dataset"]["nslkdd"]["mean_recovery_delta_pp"] for o in owners]
    rec_unswnb = [per_person[o]["defense"]["per_dataset"]["unswnb15"]["mean_recovery_delta_pp"] for o in owners]

    # Defense Side-Effects (Detection on Clean Traffic)
    clean_baseline = [per_person[o]["defense"]["overall"]["mean_clean_baseline_detection_rate"] * 100 for o in owners]
    clean_adversarial = [per_person[o]["defense"]["overall"]["mean_clean_adversarial_detection_rate"] * 100 for o in owners]

    # 3. Build HTML Table Rows
    owner_table_html = ""
    for o in owners:
        d = per_person[o]
        owner_table_html += f"""
        <tr>
            <td class="fw-bold">{o.capitalize()}</td>
            <td>{d['attacks']['attack_a']['overall_mean_esr']:.4f}</td>
            <td>{d['attacks']['attack_b']['overall_mean_esr']:.4f}</td>
            <td>{d['attacks']['attack_c']['overall_mean_esr']:.4f}</td>
            <td class="table-primary fw-bold">{d['defense']['overall']['mean_recovery_delta_pp']:.2f} PP</td>
            <td class="table-primary">{d['defense']['overall']['target_hit_rate_overall'] * 100:.1f}%</td>
            <td class="table-warning">{d['adv_training_clean']['mean_accuracy_drop_pp']:.4f} PP</td>
            <td class="table-success fw-bold">{d['retrained_adversarial']['overall_mean_accuracy']:.4f}</td>
        </tr>
        """

    # 4. Generate the HTML String (Injecting JSON dumps into the JS)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Team Metrics Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ background-color: #f8f9fc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .card {{ border: none; border-radius: 10px; box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15); margin-bottom: 24px; }}
        .card-header {{ background-color: #f8f9fc; border-bottom: 1px solid #e3e6f0; font-weight: bold; color: #4e73df; }}
        .card-header.bg-defense {{ background-color: #1cc88a; color: white; }}
        .card-header.bg-attack {{ background-color: #e74a3b; color: white; }}
        .chart-wrapper {{ height: 320px; position: relative; padding: 15px; }}
        .top-banner {{ background: linear-gradient(180deg, #4e73df 10%, #224abe 100%); color: white; padding: 30px 0; margin-bottom: 30px; }}
        .section-divider {{ margin: 40px 0 20px 0; font-weight: 700; color: #5a5c69; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid #e3e6f0; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="top-banner text-center">
        <h2 class="fw-bold">Adversarial ML Team Metrics Dashboard</h2>
        <p class="mb-0">Static Generation Data Timestamp (UTC): {generated_at}</p>
    </div>

    <div class="container-fluid px-5">
        
        <h4 class="section-divider">Phase 1: Overall Evasion & Retraining Summary</h4>
        
        <div class="row">
            <div class="col-xl-6 col-lg-6">
                <div class="card">
                    <div class="card-header bg-attack">Attack Evasion Success Rate (ESR) by Owner</div>
                    <div class="card-body chart-wrapper">
                        <canvas id="attackChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="col-xl-6 col-lg-6">
                <div class="card">
                    <div class="card-header">Retrained Accuracy by Model Architecture</div>
                    <div class="card-body chart-wrapper">
                        <canvas id="modelChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <h4 class="section-divider">Phase 2: Deep Dive into Defense Efficacy</h4>

        <div class="row">
            <div class="col-xl-4 col-lg-6">
                <div class="card">
                    <div class="card-header bg-defense">Defense Recovery vs. Hit Rate</div>
                    <div class="card-body chart-wrapper">
                        <canvas id="defenseComboChart"></canvas>
                    </div>
                    <div class="card-footer small text-muted text-center">
                        Bar: Avg Recovery (PP) | Line: Target Hit Rate (%)
                    </div>
                </div>
            </div>

            <div class="col-xl-4 col-lg-6">
                <div class="card">
                    <div class="card-header bg-defense">Defense Recovery Delta by Dataset</div>
                    <div class="card-body chart-wrapper">
                        <canvas id="datasetRecoveryChart"></canvas>
                    </div>
                    <div class="card-footer small text-muted text-center">
                        Shows which datasets were easiest/hardest for defenses to recover accuracy on.
                    </div>
                </div>
            </div>

            <div class="col-xl-4 col-lg-12">
                <div class="card">
                    <div class="card-header bg-defense">Defense Impact on Clean Traffic Detection</div>
                    <div class="card-body chart-wrapper">
                        <canvas id="cleanTrafficChart"></canvas>
                    </div>
                    <div class="card-footer small text-muted text-center">
                        Compares baseline clean detection vs. adversarial clean detection (identifies false positives cost).
                    </div>
                </div>
            </div>
        </div>

        <h4 class="section-divider">Phase 3: Detailed Data Matrix</h4>

        <div class="row mb-5">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">Comprehensive Owner Performance Matrix</div>
                    <div class="card-body p-0 table-responsive">
                        <table class="table table-hover table-striped mb-0 text-center align-middle">
                            <thead class="table-dark">
                                <tr>
                                    <th>Owner</th>
                                    <th>ESR (Attack A)</th>
                                    <th>ESR (Attack B)</th>
                                    <th>ESR (Attack C)</th>
                                    <th>Recovery Delta (PP)</th>
                                    <th>Target Hit Rate</th>
                                    <th>Training Acc Drop</th>
                                    <th>Final Retrained Acc</th>
                                </tr>
                            </thead>
                            <tbody>
                                {owner_table_html}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data injected statically via Python JSON dumps
        const owners = {json.dumps(formatted_owners)};
        
        // 1. Attack ESR Chart
        new Chart(document.getElementById('attackChart'), {{
            type: 'bar',
            data: {{
                labels: owners,
                datasets: [
                    {{ label: 'Attack A', data: {json.dumps(esr_a)}, backgroundColor: '#e74a3b' }},
                    {{ label: 'Attack B', data: {json.dumps(esr_b)}, backgroundColor: '#f6c23e' }},
                    {{ label: 'Attack C', data: {json.dumps(esr_c)}, backgroundColor: '#e83e8c' }}
                ]
            }},
            options: {{ maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true, max: 1, title: {{ display: true, text: 'Mean ESR' }} }} }} }}
        }});

        // 2. Retrained Models Chart
        new Chart(document.getElementById('modelChart'), {{
            type: 'bar',
            data: {{
                labels: owners,
                datasets: [
                    {{ label: 'MLP', data: {json.dumps(model_acc['mlp'])}, backgroundColor: '#4e73df' }},
                    {{ label: 'Random Forest', data: {json.dumps(model_acc['random_forest'])}, backgroundColor: '#36b9cc' }},
                    {{ label: 'XGBoost', data: {json.dumps(model_acc['xgboost'])}, backgroundColor: '#f6c23e' }},
                    {{ label: 'LSTM', data: {json.dumps(model_acc['lstm'])}, backgroundColor: '#5a5c69' }}
                ]
            }},
            options: {{ maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: false, min: 0.6, title: {{ display: true, text: 'Accuracy' }} }} }} }}
        }});

        // 3. Defense Combo Chart (Recovery PP vs Hit Rate %)
        new Chart(document.getElementById('defenseComboChart'), {{
            type: 'bar',
            data: {{
                labels: owners,
                datasets: [
                    {{
                        type: 'line',
                        label: 'Target Hit Rate (%)',
                        data: {json.dumps(hit_rate_overall)},
                        borderColor: '#f6c23e',
                        backgroundColor: '#f6c23e',
                        borderWidth: 2,
                        yAxisID: 'y1',
                        tension: 0.3
                    }},
                    {{
                        type: 'bar',
                        label: 'Avg Recovery Delta (PP)',
                        data: {json.dumps(recovery_overall)},
                        backgroundColor: '#1cc88a',
                        yAxisID: 'y'
                    }}
                ]
            }},
            options: {{
                maintainAspectRatio: false,
                scales: {{
                    y: {{ type: 'linear', display: true, position: 'left', title: {{ display: true, text: 'Recovery (PP)' }} }},
                    y1: {{ type: 'linear', display: true, position: 'right', grid: {{ drawOnChartArea: false }}, min: 0, max: 100, title: {{ display: true, text: 'Hit Rate (%)' }} }}
                }}
            }}
        }});

        // 4. Recovery by Dataset Chart
        new Chart(document.getElementById('datasetRecoveryChart'), {{
            type: 'bar',
            data: {{
                labels: owners,
                datasets: [
                    {{ label: 'CICIDS-2017', data: {json.dumps(rec_cicids)}, backgroundColor: '#4e73df' }},
                    {{ label: 'NSL-KDD', data: {json.dumps(rec_nslkdd)}, backgroundColor: '#36b9cc' }},
                    {{ label: 'UNSW-NB15', data: {json.dumps(rec_unswnb)}, backgroundColor: '#1cc88a' }}
                ]
            }},
            options: {{ maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Recovery Delta (PP)' }} }} }} }}
        }});

        // 5. Clean Traffic Side Effects (Radar or Bar)
        new Chart(document.getElementById('cleanTrafficChart'), {{
            type: 'bar',
            data: {{
                labels: owners,
                datasets: [
                    {{ label: 'Baseline Clean Detection (%)', data: {json.dumps(clean_baseline)}, backgroundColor: '#5a5c69' }},
                    {{ label: 'Adversarial Clean Detection (%)', data: {json.dumps(clean_adversarial)}, backgroundColor: '#1cc88a' }}
                ]
            }},
            options: {{ 
                maintainAspectRatio: false, 
                scales: {{ y: {{ min: 95, max: 100, title: {{ display: true, text: 'Detection Rate (%)' }} }} }} 
            }}
        }});
    </script>
</body>
</html>"""

    # 5. Write the static HTML file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Success! Hardcoded dashboard generated at: {output_path}")

if __name__ == "__main__":
    generate_static_dashboard()
import json
import os

def generate_comprehensive_dashboard(json_path="results/team_metrics_summary.json", output_path="results/results_summary_dashboard.html"):
    # 1. Dynamically pull data from the JSON file
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}. Please ensure it is in the same directory.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    owners = data.get("owners", [])
    per_person = data.get("per_person_summary", {})

    # 2. Extract original metrics for earlier visualizations
    esr_a = [per_person[o]["attacks"]["attack_a"]["overall_mean_esr"] for o in owners]
    esr_b = [per_person[o]["attacks"]["attack_b"]["overall_mean_esr"] for o in owners]
    esr_c = [per_person[o]["attacks"]["attack_c"]["overall_mean_esr"] for o in owners]

    models = ["mlp", "random_forest", "xgboost", "lstm", "majority_voting"]
    model_acc = {m: [per_person[o]["retrained_adversarial"]["model_mean_accuracy"][m] for o in owners] for m in models}

    datasets = ["cicids2017", "nslkdd", "unswnb15"]
    radar_datasets = []
    colors = ['#ff6384', '#36a2eb', '#ffce56']
    
    for i, o in enumerate(owners):
        radar_datasets.append({
            "label": o.capitalize(),
            "data": [per_person[o]["retrained_adversarial"]["dataset_mean_accuracy"][ds] for ds in datasets],
            "borderColor": colors[i % len(colors)],
            "backgroundColor": colors[i % len(colors)] + "33", # 20% opacity
            "pointBackgroundColor": colors[i % len(colors)],
            "fill": True
        })

    # 3. Aggregations for Defense & Attack Breakdown (New Section)
    # Team Average Attack ESR
    avg_esr_a = sum(esr_a) / len(esr_a) if esr_a else 0
    avg_esr_b = sum(esr_b) / len(esr_b) if esr_b else 0
    avg_esr_c = sum(esr_c) / len(esr_c) if esr_c else 0
    team_avg_attacks = [avg_esr_a, avg_esr_b, avg_esr_c]

    # Team Average Defense Recovery by Dataset
    defense_rec_cicids = [per_person[o]["defense"]["per_dataset"]["cicids2017"]["mean_recovery_delta_pp"] for o in owners]
    defense_rec_nslkdd = [per_person[o]["defense"]["per_dataset"]["nslkdd"]["mean_recovery_delta_pp"] for o in owners]
    defense_rec_unswnb = [per_person[o]["defense"]["per_dataset"]["unswnb15"]["mean_recovery_delta_pp"] for o in owners]
    
    avg_rec_cicids = sum(defense_rec_cicids) / len(defense_rec_cicids) if defense_rec_cicids else 0
    avg_rec_nslkdd = sum(defense_rec_nslkdd) / len(defense_rec_nslkdd) if defense_rec_nslkdd else 0
    avg_rec_unswnb = sum(defense_rec_unswnb) / len(defense_rec_unswnb) if defense_rec_unswnb else 0
    team_avg_recovery = [avg_rec_cicids, avg_rec_nslkdd, avg_rec_unswnb]

    # Individual Target Hit Rates
    target_hit_rates = [per_person[o]["defense"]["overall"]["target_hit_rate_overall"] for o in owners]

    # 4. Build HTML Table Rows
    owner_table_html = ""
    for o in owners:
        d = per_person[o]
        owner_table_html += f"""
        <tr>
            <td class="fw-bold">{o.capitalize()}</td>
            <td>{d['attacks']['attack_a']['overall_mean_esr']:.4f}</td>
            <td>{d['attacks']['attack_b']['overall_mean_esr']:.4f}</td>
            <td>{d['attacks']['attack_c']['overall_mean_esr']:.4f}</td>
            <td class="table-primary fw-bold">{d['defense']['overall']['mean_recovery_delta_pp']:.2f}%</td>
            <td class="table-primary">{d['defense']['overall']['target_hit_rate_overall']:.2f}</td>
            <td class="table-warning">{d['adv_training_clean']['mean_accuracy_drop_pp']:.4f}%</td>
            <td class="table-success fw-bold">{d['retrained_adversarial']['overall_mean_accuracy']:.4f}</td>
        </tr>
        """

    # 5. Generate the final HTML with Bootstrap & Chart.js
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Team Metrics</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ background-color: #f8f9fc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .card {{ border: none; border-radius: 10px; box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15); margin-bottom: 24px; }}
            .card-header {{ background-color: #f8f9fc; border-bottom: 1px solid #e3e6f0; font-weight: bold; color: #4e73df; }}
            .card-header.bg-defense {{ background-color: #1cc88a; color: white; }}
            .card-header.bg-attack {{ background-color: #e74a3b; color: white; }}
            .chart-wrapper {{ height: 300px; position: relative; }}
            .top-banner {{ background: linear-gradient(180deg, #4e73df 10%, #224abe 100%); color: white; padding: 30px 0; margin-bottom: 30px; }}
            .section-divider {{ margin: 40px 0 20px 0; font-weight: 700; color: #5a5c69; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid #e3e6f0; padding-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="top-banner text-center">
            <h2 class="fw-bold">Adversarial ML Team Metrics Dashboard</h2>
            <p class="mb-0">Generated from dynamic JSON source | UTC: {data.get('generated_at_utc', 'N/A')}</p>
        </div>

        <div class="container-fluid px-5">
            
            <h4 class="section-divider">Phase 1: Overall Evasion & Retraining Performance</h4>
            
            <div class="row">
                <div class="col-xl-4 col-lg-6">
                    <div class="card">
                        <div class="card-header">Attack Evasion Success Rate (ESR) per Owner</div>
                        <div class="card-body chart-wrapper">
                            <canvas id="attackChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="col-xl-4 col-lg-6">
                    <div class="card">
                        <div class="card-header">Retrained Accuracy by Model Architecture</div>
                        <div class="card-body chart-wrapper">
                            <canvas id="modelChart"></canvas>
                        </div>
                    </div>
                </div>

                <div class="col-xl-4 col-lg-12">
                    <div class="card">
                        <div class="card-header">Overall Retrained Accuracy by Dataset</div>
                        <div class="card-body chart-wrapper d-flex justify-content-center">
                            <canvas id="datasetChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <h4 class="section-divider">Phase 2: Attack vs. Defense Efficacy Breakdown</h4>

            <div class="row">
                <div class="col-xl-4 col-lg-6">
                    <div class="card">
                        <div class="card-header bg-attack">Most Successful Attacks (Team Avg ESR)</div>
                        <div class="card-body chart-wrapper">
                            <canvas id="avgAttackChart"></canvas>
                        </div>
                        <div class="card-footer small text-muted text-center">
                            Identifies which attack generation method bypassed defenses most frequently.
                        </div>
                    </div>
                </div>

                <div class="col-xl-4 col-lg-6">
                    <div class="card">
                        <div class="card-header bg-defense">Defense Recovery Delta per Dataset</div>
                        <div class="card-body chart-wrapper">
                            <canvas id="defenseRecoveryChart"></canvas>
                        </div>
                        <div class="card-footer small text-muted text-center">
                            Identifies which datasets were easiest (high delta) or hardest (low delta) to recover accuracy for.
                        </div>
                    </div>
                </div>

                <div class="col-xl-4 col-lg-12">
                    <div class="card">
                        <div class="card-header bg-defense">Defense Target Hit Rate per Owner</div>
                        <div class="card-body chart-wrapper">
                            <canvas id="hitRateChart"></canvas>
                        </div>
                        <div class="card-footer small text-muted text-center">
                            Proportion of evaluated mutations that successfully met the recovery threshold.
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
                                        <th>Defense Hit Rate</th>
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
            const owners = {json.dumps([o.capitalize() for o in owners])};
            
            // 1. Grouped Bar Chart for Attack ESRs per owner
            new Chart(document.getElementById('attackChart'), {{
                type: 'bar',
                data: {{
                    labels: owners,
                    datasets: [
                        {{ label: 'Attack A', data: {json.dumps(esr_a)}, backgroundColor: '#e74a3b' }},
                        {{ label: 'Attack B', data: {json.dumps(esr_b)}, backgroundColor: '#f6c23e' }},
                        {{ label: 'Attack C', data: {json.dumps(esr_c)}, backgroundColor: '#1cc88a' }}
                    ]
                }},
                options: {{ maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true, max: 1 }} }} }}
            }});

            // 2. Bar Chart for Model Architectures
            new Chart(document.getElementById('modelChart'), {{
                type: 'bar',
                data: {{
                    labels: owners,
                    datasets: [
                        {{ label: 'MLP', data: {json.dumps(model_acc['mlp'])}, backgroundColor: '#4e73df' }},
                        {{ label: 'Random Forest', data: {json.dumps(model_acc['random_forest'])}, backgroundColor: '#36b9cc' }},
                        {{ label: 'XGBoost', data: {json.dumps(model_acc['xgboost'])}, backgroundColor: '#858796' }},
                        {{ label: 'LSTM', data: {json.dumps(model_acc['lstm'])}, backgroundColor: '#5a5c69' }}
                    ]
                }},
                options: {{ maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: false, min: 0.6 }} }} }}
            }});

            // 3. Radar Chart for Datasets
            new Chart(document.getElementById('datasetChart'), {{
                type: 'radar',
                data: {{
                    labels: ['CICIDS-2017', 'NSL-KDD', 'UNSW-NB15'],
                    datasets: {json.dumps(radar_datasets)}
                }},
                options: {{ 
                    maintainAspectRatio: false,
                    scales: {{ r: {{ min: 0.8, max: 1.0, ticks: {{ stepSize: 0.05 }} }} }} 
                }}
            }});

            // 4. Team Average Attack Success
            new Chart(document.getElementById('avgAttackChart'), {{
                type: 'bar',
                data: {{
                    labels: ['Attack A', 'Attack B', 'Attack C'],
                    datasets: [{{
                        label: 'Team Average ESR',
                        data: {json.dumps(team_avg_attacks)},
                        backgroundColor: ['#e74a3b', '#e74a3b', '#e74a3b'],
                        borderWidth: 1
                    }}]
                }},
                options: {{ 
                    maintainAspectRatio: false, 
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Evasion Success Rate' }} }} }} 
                }}
            }});

            // 5. Defense Recovery by Dataset
            new Chart(document.getElementById('defenseRecoveryChart'), {{
                type: 'bar',
                data: {{
                    labels: ['CICIDS-2017', 'NSL-KDD', 'UNSW-NB15'],
                    datasets: [{{
                        label: 'Average Recovery Delta (PP)',
                        data: {json.dumps(team_avg_recovery)},
                        backgroundColor: ['#1cc88a', '#1cc88a', '#1cc88a'],
                        borderWidth: 1
                    }}]
                }},
                options: {{ 
                    maintainAspectRatio: false, 
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Percentage Points (PP)' }} }} }} 
                }}
            }});

            // 6. Target Hit Rate per Owner
            new Chart(document.getElementById('hitRateChart'), {{
                type: 'doughnut',
                data: {{
                    labels: owners,
                    datasets: [{{
                        data: {json.dumps(target_hit_rates)},
                        backgroundColor: ['#4e73df', '#1cc88a', '#36b9cc'],
                        hoverOffset: 4
                    }}]
                }},
                options: {{ 
                    maintainAspectRatio: false,
                    cutout: '60%',
                    plugins: {{ 
                        tooltip: {{ callbacks: {{ label: function(context) {{ return ' Hit Rate: ' + context.parsed + '%'; }} }} }} 
                    }} 
                }}
            }});
        </script>
    </body>
    </html>
    """

    # 6. Write the final string to an HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Success! Dynamic dashboard generated at: {output_path}")

if __name__ == "__main__":
    generate_comprehensive_dashboard()
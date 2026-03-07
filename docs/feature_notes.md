# Feature Constraint Register

# Person 1 | Sprint 1 deliverable | Last updated: post-EDA validation

# Consumers: Category A (Person 2), Category B (Person 3), Category C (Person 1)

---

## How to use this document

Before implementing any attack perturbation, check three things:

1. Is the feature in the MUTABLE list for your dataset? If not, do not touch it.
2. Does it have INTERDEPENDENCIES? If yes, update all dependent features consistently.
3. Does it have BOUNDS? Clamp your perturbed value to the stated range.

---

## CICIDS2017

### Features confirmed dropped (constant = always 0, absent from .npz arrays)

Bwd PSH Flags, Bwd URG Flags,
Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate,
Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate

### ✅ Mutable features (safe to perturb independently)

- Flow Duration — clip to [0, inf] (raw data has min=-13, CICFlowMeter bug)
- Total Fwd Packets
- Total Backward Packets
- Total Length of Fwd Packets
- Total Length of Bwd Packets
- Flow IAT Mean / Std / Max / Min
- Fwd IAT Total / Mean / Std / Max / Min
- Bwd IAT Total / Mean / Std / Max / Min
- Fwd Packet Length Max / Min / Mean / Std
- Bwd Packet Length Max / Min / Mean / Std
- Min / Max / Mean / Std Packet Length
- Fwd PSH Flags — confirmed non-constant (retained in arrays)
- FIN Flag Count / SYN Flag Count / RST Flag Count / PSH Flag Count
- Init_Win_bytes_forward — bounds: [0, 65535]
- Init_Win_bytes_backward — bounds: [0, 65535]
- act_data_pkt_fwd — bounds: [0, Total Fwd Packets]
- min_seg_size_forward — bounds: [20, inf]
- Active Mean / Std / Max / Min
- Idle Mean / Std / Max / Min

### ❌ Immutable features (derived — mutating independently violates physics)

| Feature                    | Why                                   | Constraint                                        |
| -------------------------- | ------------------------------------- | ------------------------------------------------- |
| Flow Bytes/s               | Derived (had 1509 Inf values from ÷0) | = (fwd_bytes + bwd_bytes) / flow_duration         |
| Flow Packets/s             | Derived (had 2867 Inf values from ÷0) | = (fwd_pkts + bwd_pkts) / flow_duration           |
| Average Packet Size        | Derived                               | = (fwd_bytes + bwd_bytes) / (fwd_pkts + bwd_pkts) |
| Avg Fwd Segment Size       | Derived                               | ≈ fwd_bytes / fwd_pkts                            |
| Avg Bwd Segment Size       | Derived                               | ≈ bwd_bytes / bwd_pkts                            |
| Packet Length Variance     | Derived                               | = Packet Length Std²                              |
| Fwd Packets/s              | Derived                               | = fwd_pkts / flow_duration                        |
| Bwd Packets/s              | Derived                               | = bwd_pkts / flow_duration                        |
| Down/Up Ratio              | Derived                               | = bwd_bytes / fwd_bytes                           |
| Subflow Fwd/Bwd Bytes/Pkts | Derived (subsets of main flow)        | ≤ corresponding main flow value                   |

### ⚠️ Conditionally mutable (protocol hard constraints)

| Feature           | Constraint                                                         |
| ----------------- | ------------------------------------------------------------------ |
| ACK Flag Count    | Must be ≥ SYN Flag Count in established connections                |
| FIN + RST         | Mutually exclusive — both > 0 in same flow is invalid              |
| PSH Flag Count    | Only valid when payload bytes > 0                                  |
| Fwd Header Length | 20 × Total_Fwd_Packets ≤ value ≤ 60 × Total_Fwd_Packets            |
| SYN Flag Count    | Typically 1 per flow; > 1 indicates SYN flood (may be intentional) |

### Class imbalance — attack target guidance

- BENIGN is 80.3% — baseline models will be biased toward predicting BENIGN
- Recommended primary targets: DoS Hulk (231k), PortScan (158k), DDoS (128k)
- Avoid as primary targets: Heartbleed (11), SQL Injection (21), Infiltration (36)
- Interpret carefully: PortScan, Bot, DoS Hulk (known labeling/stat errors)

---

## NSL-KDD

### Feature confirmed dropped (constant = always 0)

num_outbound_cmds — confirmed constant, absent from .npz arrays (39 → 38 features)

### ✅ Mutable features

duration, src_bytes, dst_bytes, count, srv_count,
serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate,
same_srv_rate, diff_srv_rate, srv_diff_host_rate,
dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,
dst_host_diff_srv_rate, dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate, dst_host_serror_rate,
dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate,
hot, num_failed_logins, num_compromised, root_shell, su_attempted,
num_root, num_file_creations, num_shells, num_access_files,
land, wrong_fragment, urgent

### ❌ Immutable features

- protocol_type, service, flag — categorical; changing breaks attack semantics
- is_host_login, is_guest_login, logged_in — binary semantic flags

### ⚠️ Class imbalance warning — CRITICAL

R2L = 0.79%, U2R = 0.04% of total data.
A model can score >99% accuracy while never predicting either class.
→ Always report per-class F1, never just overall accuracy.
→ Always use class_weight='balanced' in all models.
→ Adversarial examples targeting R2L/U2R will be statistically unreliable.
Recommended attack targets: DoS (36.46%), Probe (9.25%)

---

## UNSW-NB15

### Features confirmed dropped (constant = always 0)

Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, URG Flag Count,
CWR Flag Count, ECE Flag Count,
Fwd Bytes/Bulk Avg, Fwd Packet/Bulk Avg, Fwd Bulk Rate Avg

### ✅ Mutable features

dur, sbytes, dbytes, Spkts, Dpkts,
sttl*, dttl*, sloss, dloss,
swin, dwin, smeansz, dmeansz,
Sjit, Djit, Sintpkt, Dintpkt,
tcprtt, synack, ackdat,
trans_depth, res_bdy_len,
ct_state_ttl, ct_srv_src, ct_srv_dst,
ct_dst_ltm, ct_src_ltm,
ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm,
is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd

\*sttl/dttl are OS-level TTL values — mutable for feature-space attacks
but physically unrealistic to change (document this limitation)

### ❌ Immutable features

- proto, state, service — categorical protocol identifiers
- is_sm_ips_ports — binary semantic flag (same src/dst IP:port)
- stcpb, dtcpb — TCP base sequence numbers (OS fingerprint)
- Sload, Dload — bits/sec (derived from sbytes/dbytes + dur)

### ⚠️ Class imbalance warning

Normal is 80%. Worms (0.05%), Fuzzers (0.09%), Analysis (0.10%) too sparse.
→ Use class_weight='balanced'. Report per-class F1.
Recommended attack targets: DoS (6.91%), Exploits (6.61%), Reconnaissance (3.74%)

### Label structure note

Integer labels 0–9 stored in separate Label.csv, joined on row index.
All 10 labels mapped successfully — no unmapped values confirmed by EDA.
31.64% duplicates remov

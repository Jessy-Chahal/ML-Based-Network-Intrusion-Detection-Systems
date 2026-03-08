# Functional Preservation Thresholds

DOS_PKT_RATE_MIN_RATIO = 0.70 # flow_pkts_s must retain 70% of original
PORTSCAN_PKT_RATE_MIN_RATIO = 0.60 # proxy for 80% port coverage (no per-port data in CICIDS2017)
C2_DURATION_MAX_RATIO = 2.0 # flow_duration cannot exceed 2x original
C2_IAT_MEAN_MAX_RATIO = 3.0 # flow_iat_mean cannot exceed 3x original (Bot flows only)
BRUTEFORCE_PKT_RATE_MIN_RATIO = 0.60 # flow_pkts_s must retain 60% of original

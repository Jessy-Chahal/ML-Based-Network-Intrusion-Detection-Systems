# TCP Protocol Validity Thresholds

```
MIN_PKT_BYTES = 20 # Min IP header size (bytes) - protocol hard floor

MAX_PKT_BYTES = 65535 # Max IP payload (bytes) - protocol hard ceiling

MIN_TCP_HEADER = 20 # Min TCP header length (bytes)

FLOW_DURATION_MIN = 0 # Flow duration cannot be negative
```

# DNS Protocol Validity Thresholds

```
DNS_PORT = 53 # Expected destination port for DNS flows

MAX_DNS_PAYLOAD_BYTES = 4096 # EDNS0 max for both query and response packets

MAX_DNS_DURATION_US = 5000000 # DNS resolver timeout ceiling (microseconds)

MAX_DNS_FWD_PKTS = 10 # Max forward packets before flow is anomalous
```

# Functional Preservation Thresholds

```
DOS_PKT_RATE_MIN_RATIO = 0.70 # flow_pkts_s must retain 70% of original

PORTSCAN_PKT_RATE_MIN_RATIO = 0.60 # proxy for 80% port coverage (no per-port data in CICIDS2017)

C2_DURATION_MAX_RATIO = 2.0 # flow_duration cannot exceed 2x original

C2_IAT_MEAN_MAX_RATIO = 3.0 # flow_iat_mean cannot exceed 3x original (Bot flows only)

BRUTEFORCE_PKT_RATE_MIN_RATIO = 0.60 # flow_pkts_s must retain 60% of original
```

# Behavioural Plausibility Thresholds

```
MAX_IAT_MEAN_S = 60 # flow_iat_mean ceiling (seconds) - stateful firewall timeout

MAX_FLOW_PKTS_S = 1000000 # flow_pkts_per_sec ceiling - physical NIC limit

MIN_AVG_PKT_BYTES = 20 # avg packet size floor - header-only floor

MAX_AVG_PKT_BYTES = 1500 # avg packet size ceiling - standard Ethernet MTU

```

# Attacker Capability

```
No numeric thresholds. Type 4 is enforced entirely through code review and the MutationRegistry. There is nothing to express as constant.
```

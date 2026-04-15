# Docker Quick Start Guide

This project is containerized so an evaluator can run your exact code, models, and data with minimal setup.

## What Is Included In-Container

The Docker image includes the full repository contents needed for evaluation, including:

- `data/`
- `docs/`
- `models/`
- `notebooks/`
- `results/`
- `scripts/`
- `src/`
- `tests/`

The Compose setup also bind-mounts the repo (`.:/workspace`) so new outputs are persisted on the host.

## Prerequisites

- Docker 20.10+
- Docker Compose V2 (`docker compose`)

## Quick Start (Recommended)

```bash
# Build image and start the dev container
docker compose up -d --build

# Open a shell in the container
docker compose exec nids bash

# Run tests
docker compose exec nids pytest

# Run a script
docker compose exec nids python scripts/summarize_team_metrics.py
```

## Jupyter

```bash
docker compose exec nids jupyter lab --ip=0.0.0.0 --no-browser
```

Then open `http://localhost:8888`.

## Running Without Compose

```bash
# Build
docker build -t nids-adv:latest .

# Interactive shell with host persistence
docker run --rm -it -v "$(pwd):/workspace" -p 8888:8888 nids-adv:latest
```

## Persistence Behavior

Because the repo is mounted into `/workspace` in Compose:

- New files in `results/` persist on the host.
- New model files in `models/` persist on the host.
- Code edits from either side remain visible.

## Useful Commands

```bash
# Show container logs
docker compose logs nids

# Stop container
docker compose down

# Stop + remove containers/networks/volumes
docker compose down -v

# Stop + remove images
docker compose down --rmi all -v

# Remove dangling build cache files
docker builder prune -a
```

## Notes

- Container runs as non-root user `researcher`.
- Python/dependencies come from `environment.yml` (conda env `nids-adv`).
- If you need root shell for troubleshooting:

```bash
docker compose exec -u root nids bash
```

#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/sam_vit_b.yaml}
python -m waterseg.cli.prepare --config "$CONFIG"
python -m waterseg.cli.train --config "$CONFIG"
python -m waterseg.cli.evaluate --config "$CONFIG"

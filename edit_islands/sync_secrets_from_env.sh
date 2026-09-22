#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/genai_image_editing/.env"
OUT="$ROOT/edit_islands/EditIslands/Secrets.plist"
python3 - "$ENV_FILE" "$OUT" <<'PY'
import sys, plistlib
from pathlib import Path
env_path, out = Path(sys.argv[1]), Path(sys.argv[2])
keys = {}
for line in env_path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    k, v = k.strip(), v.strip().strip('"').strip("'")
    if v:
        keys[k] = v
out.write_bytes(plistlib.dumps(keys, fmt=plistlib.FMT_XML))
print(f"Synced {len(keys)} entries → {out}")
PY

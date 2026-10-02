#!/usr/bin/env bash
# Run the weekend-hunt CLI in the ml-trading env, from any directory:
#     bash scripts/hunt/hunt.sh <cmd> [args]
# HUNT_ENV overrides the env root.
set -euo pipefail

ENV_ROOT="${HUNT_ENV:-$USERPROFILE/miniforge3/envs/ml-trading}"
ENV_ROOT="$(cygpath -u "$ENV_ROOT")"
if [ ! -x "$ENV_ROOT/python.exe" ]; then
    echo "hunt.sh: no python.exe under $ENV_ROOT (set HUNT_ENV)" >&2
    exit 1
fi

# The env's DLL dirs MUST lead PATH: without them numpy's LAPACK fails to load and the
# process dies with exit 127 and no traceback.
export PATH="$ENV_ROOT:$ENV_ROOT/Library/bin:$ENV_ROOT/Library/mingw-w64/bin:$ENV_ROOT/Library/usr/bin:$ENV_ROOT/Scripts:$PATH"
export PYTHONIOENCODING=utf-8

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec python -m src.stock_screener.hunt "$@"

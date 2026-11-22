#!/bin/bash
# Run commands inside the project venv:  wsl -d Debian -- bash tools/wsl_run.sh <cmd...>
source ~/venvs/pnp/bin/activate
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
"$@"

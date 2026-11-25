#!/bin/bash
# Validation runs.  Usage: bash tools/validate.sh
python tools/check_transient.py | tail -2
python tools/check_case_bcs.py | tail -1
for c in gouy_chapman eo_slit mms charged_sphere; do
  echo "=== $c"
  python run.py "$c" --name "val_$c" 2>&1 | grep -v "^  it" | tail -16
done

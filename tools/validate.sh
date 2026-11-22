#!/bin/bash
# Validation runs.  Usage: bash tools/validate.sh
for c in gouy_chapman eo_slit mms charged_sphere; do
  echo "=== $c"
  python run.py "$c" --name "val_$c" 2>&1 | grep -v "^  it" | tail -16
done

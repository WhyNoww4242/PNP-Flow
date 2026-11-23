"""Export run profiles to CSV (results/<run>/csv/).  See pnpflow/export.py.

    python tools/export_csv.py results/pore_narrow [results/other_run ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnpflow.export import export_csv  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for folder in sys.argv[1:]:
        names = export_csv(folder)
        print(f"{folder}/csv: {len(names)} files")
        for n in names:
            print("  " + n)

"""Write fields_time.json for an existing time-dependent run from its solution.npz.

    python tools/export_fields_time.py results/<run> [...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pnpflow.output import _round, write_json  # noqa: E402


def export(folder, digits=5):
    s = json.load(open(os.path.join(folder, "summary.json")))
    z = np.load(os.path.join(folder, "solution.npz"))
    species = s["model"]["species"]
    S, n = len(species), z["X"].shape[0]
    t_per = float(s.get("transient", {}).get("t_per_tau", 1.0))
    U_t, t = z["U_t"], z["t"]
    fields = {sp["name"]: [_round(U[i * n:(i + 1) * n], digits) for U in U_t]
              for i, sp in enumerate(species)}
    if S:
        fields["rho"] = [_round(sum(sp["z"] * U[i * n:(i + 1) * n] for i, sp in enumerate(species)), digits)
                         for U in U_t]
    fields["phi"] = [_round(U[S * n:(S + 1) * n], digits) for U in U_t]
    write_json(os.path.join(folder, "fields_time.json"),
               {"t": [float(x) for x in t], "tau": [float(x) / t_per for x in t],
                "t_per_tau": t_per, "n_nodes": int(n), "fields": fields})
    print(f"{folder}: {len(t)} times, {os.path.getsize(os.path.join(folder, 'fields_time.json')) / 1e6:.1f} MB")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        export(f)

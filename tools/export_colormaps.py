"""Export cmasher colormaps as 256-entry look-up tables for the web UI.

    python tools/export_colormaps.py [out.json]

Writes app/ui/data/colormaps.json:
    {name: {"type": "sequential" | "diverging", "colors": [256 "#rrggbb"]}}
The UI maps a normalised value t in [0, 1] to colors[round(255 t)]; for a
diverging map, t = 0.5 is the value 0.
"""
import json
import os
import sys

import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "app", "ui", "data", "colormaps.json")

# name -> use (documentation only; the type comes from cmasher)
# Only these three maps are used anywhere (never viridis or other matplotlib defaults).
MAPS = {
    "cosmic": "default: potential, concentrations, mesh quality",
    "ember": "magnitudes: |E|, |u|, vorticity",
    "bubblegum": "charge density: rho, rho_f",
}


def lut(name, n=256):
    cm = plt.get_cmap("cmr." + name)
    rgb = (np.asarray(cm(np.linspace(0.0, 1.0, n)))[:, :3] * 255).round().astype(int)
    return ["#%02x%02x%02x" % tuple(c) for c in rgb]


def main(out=OUT):
    data = {}
    for name, use in MAPS.items():
        kind = cmr.get_cmap_type("cmr." + name)
        data[name] = {"type": kind, "use": use, "colors": lut(name)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {len(data)} colormaps to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])

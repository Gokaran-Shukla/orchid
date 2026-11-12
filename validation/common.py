import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.figsize": (5.2, 3.8), "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
C = ["#1f5fa8", "#d1495b", "#2e933c", "#edae49", "#6a4c93", "#00798c"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def dump(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=conv)

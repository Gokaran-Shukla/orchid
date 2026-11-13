"""V16 - parallel performance.

(a) speed-up of the three CPU parallel levels on this machine:
    Numba threads (KMC), process pool and MPI ranks (disorder ensembles of
    master-equation solves); MPI results must equal the serial ones exactly.
(b) interacting-KMC throughput vs number of carriers: dense O(N_c)
    reference engine vs Fenwick/incremental O(log N_c) engine.

Run as ``python bench_parallel.py``; the MPI part re-launches this file
with ``mpiexec -n R python bench_parallel.py --mpi-worker``.
"""
import os
import subprocess
import sys
import time
import json
import numpy as np


def me_task(seed):
    from orchid.lattice import Lattice
    from orchid.master_equation import me_linear_response
    from orchid.constants import kB_eV
    lat = Lattice(36, a=1.6e-9, seed=int(seed)).set_disorder(0.1)
    return me_linear_response(lat, 0.1 / (3 * kB_eV), 1e-4, alpha=10.0)[0]


SEEDS = list(range(8))

if "--mpi-worker" in sys.argv:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("ORCHID_USE_MPI", "1")
    from orchid.parallel import parallel_map, is_root, set_threads
    set_threads(1)
    t = time.time()
    res = parallel_map(me_task, SEEDS, backend="auto")
    if is_root():
        print(json.dumps(dict(t=time.time() - t, res=res)))
    sys.exit(0)

from common import plt, save, dump, C, HERE  # noqa: E402
from orchid.parallel import parallel_map, set_threads  # noqa: E402
from orchid.lattice import Lattice  # noqa: E402
from orchid.kmc import kmc_independent, mobility_interacting  # noqa: E402
from orchid.constants import kB_eV  # noqa: E402

if __name__ == "__main__":
    out = {}
    # ---------------- Numba threads
    lat = Lattice(40, a=1.6e-9, seed=1).set_disorder(0.1)
    W = lat.rates(300, (0.005, 0, 0), alpha=10.0)
    kmc_independent(lat, W, 4, 100, 100)
    thr = {}
    for nt in (1, 2):
        set_threads(nt)
        t = time.time()
        kmc_independent(lat, W, 64, 0, 1000000)
        thr[nt] = 64e6 / (time.time() - t)
    set_threads(os.cpu_count())
    out["threads_hops_per_s"] = thr
    # ---------------- processes
    set_threads(1)
    t = time.time()
    serial = parallel_map(me_task, SEEDS, backend="serial")
    ts = time.time() - t
    t = time.time()
    proc = parallel_map(me_task, SEEDS, backend="process", workers=2)
    tp = time.time() - t
    set_threads(os.cpu_count())
    # ---------------- MPI
    mpi = {}
    for R in (1, 2):
        r = subprocess.run(["mpiexec", "-n", str(R), sys.executable, os.path.abspath(__file__),
                            "--mpi-worker"], capture_output=True, text=True, timeout=1800)
        js = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])
        mpi[R] = js
    out.update(serial_s=ts, process2_s=tp, mpi1_s=mpi[1]["t"], mpi2_s=mpi[2]["t"],
               mpi_equals_serial=bool(np.allclose(mpi[2]["res"], serial, rtol=1e-12)),
               process_equals_serial=bool(np.allclose(proc, serial, rtol=1e-12)))
    print(out, flush=True)
    # ---------------- interacting KMC scaling
    T = 0.1 / (3 * kB_eV)
    lat = Lattice(40, a=1.6e-9, seed=2).set_disorder(0.1)
    dens = np.array([2e-4, 1e-3, 5e-3, 2e-2, 5e-2])
    thr_f, thr_d = [], []
    for p in dens:
        nc = int(p * lat.N)
        mobility_interacting(lat, T, 0.01, p, n_eq=10, n_meas=10, engine="fast")
        t = time.time()
        mobility_interacting(lat, T, 0.01, p, n_eq=0, n_meas=200000, engine="fast")
        thr_f.append(2e5 / (time.time() - t))
        nd = int(min(2e5, 4e6 / nc))
        mobility_interacting(lat, T, 0.01, p, n_eq=10, n_meas=10, engine="dense")
        t = time.time()
        mobility_interacting(lat, T, 0.01, p, n_eq=0, n_meas=nd, engine="dense")
        thr_d.append(nd / (time.time() - t))
        print(p, nc, thr_f[-1], thr_d[-1], flush=True)
    ncs = dens * lat.N
    out.update(nc=ncs, fast_ev_s=thr_f, dense_ev_s=thr_d)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    labels = ["1 thread", "2 threads", "serial", "2 processes", "MPI 1 rank", "MPI 2 ranks"]
    speed = [1.0, thr[2] / thr[1], 1.0, ts / tp, 1.0, mpi[1]["t"] / mpi[2]["t"]]
    cols = [C[0], C[0], C[1], C[1], C[2], C[2]]
    ax[0].bar(range(6), speed, color=cols)
    ax[0].axhline(2, c="k", ls=":", lw=1)
    ax[0].set_xticks(range(6))
    ax[0].set_xticklabels(labels, rotation=30, fontsize=8)
    ax[0].set_ylabel("speed-up")
    ax[0].set_title("2-core test node (ideal = 2, dotted)", fontsize=10)
    ax[1].loglog(ncs, thr_f, "o-", c=C[0], label="Fenwick + incremental (ORCHID fast)")
    ax[1].loglog(ncs, thr_d, "s-", c=C[1], label="dense recompute (reference)")
    ax[1].set_xlabel("number of carriers")
    ax[1].set_ylabel("events / s")
    ax[1].set_title(r"interacting KMC, $40^3$ sites", fontsize=10)
    ax[1].legend(fontsize=8)
    save(fig, "v16_parallel")
    dump("parallel", out)

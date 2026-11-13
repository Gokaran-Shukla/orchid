r"""Parallel execution layer for ORCHID.

ORCHID uses hybrid parallelism, from inside a node out to a whole cluster:

==========  ===========================================  =======================
level       mechanism                                    used by
==========  ===========================================  =======================
SIMD/GPU    one CUDA thread per carrier / exciton        :mod:`orchid.gpu`
            CuPy sparse CG for the master equation       ``me_linear_response``
threads     Numba ``prange`` (shared memory, no GIL)     KMC, exciton KMC
processes   ``ProcessPoolExecutor`` (single node)        :func:`parallel_map`
MPI ranks   mpi4py block-cyclic task farm (multi-node)   :func:`parallel_map`
==========  ===========================================  =======================

The outer levels distribute independent *tasks*: disorder realisations,
temperatures, fields, densities, device parameter sets or search-grid
points. Such workloads scale almost perfectly, because the tasks only
communicate when results are gathered.

Usage on a cluster (``R`` ranks x ``t`` threads)::

    export NUMBA_NUM_THREADS=t
    mpiexec -n R python my_study.py      # parallel_map(..., backend="auto")

Every rank executes the script. :func:`parallel_map` gives each rank its
slice and returns the gathered, correctly ordered list on *all* ranks; use
:func:`is_root` to guard file output.
"""
import os
import math
import numpy as np

_MPI = None


def _mpi():
    """Return the mpi4py ``MPI`` module if running under >1 ranks, else None."""
    global _MPI
    if _MPI is None:
        try:
            if not any(k in os.environ for k in ("PMI_RANK", "OMPI_COMM_WORLD_RANK",
                                                  "PMI_SIZE", "MPI_LOCALRANKID",
                                                  "SLURM_PROCID", "ORCHID_USE_MPI")):
                _MPI = False
            else:
                from mpi4py import MPI
                _MPI = MPI if MPI.COMM_WORLD.Get_size() > 1 else False
        except Exception:
            _MPI = False
    return _MPI or None


def rank():
    M = _mpi()
    return M.COMM_WORLD.Get_rank() if M else 0


def size():
    M = _mpi()
    return M.COMM_WORLD.Get_size() if M else 1


def is_root():
    return rank() == 0


def root_print(*a, **k):
    if is_root():
        print(*a, **k, flush=True)


def set_threads(n):
    """Set the number of Numba threads used by KMC kernels in this process."""
    try:
        import numba
        numba.set_num_threads(int(n))
    except Exception:
        pass


def get_threads():
    try:
        import numba
        return numba.get_num_threads()
    except Exception:
        return 1


def describe():
    """Human-readable summary of the available parallel resources."""
    from . import gpu
    return dict(mpi_ranks=size(), rank=rank(), numba_threads=get_threads(),
                cpu_count=os.cpu_count(), gpu=gpu.available())


def _block_cyclic(n, nranks, r):
    return list(range(r, n, nranks))


def parallel_map(func, items, backend="auto", workers=None, chunksize=1):
    """Apply ``func`` to every element of ``items`` in parallel.

    backend : "auto" | "serial" | "process" | "thread" | "mpi"
        "auto" picks MPI when launched under >1 ranks, otherwise processes
        if ``workers`` > 1, otherwise serial.
    Returns the list of results in input order (on every MPI rank).
    """
    items = list(items)
    if backend == "auto":
        if _mpi():
            backend = "mpi"
        elif workers is not None and workers > 1:
            backend = "process"
        else:
            backend = "serial"
    if backend == "serial":
        return [func(x) for x in items]
    if backend == "thread":
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers or os.cpu_count()) as ex:
            return list(ex.map(func, items))
    if backend == "process":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
            return list(ex.map(func, items, chunksize=chunksize))
    if backend == "mpi":
        M = _mpi()
        if not M:
            return [func(x) for x in items]
        comm = M.COMM_WORLD
        r, R = comm.Get_rank(), comm.Get_size()
        mine = _block_cyclic(len(items), R, r)
        local = [(i, func(items[i])) for i in mine]
        allres = comm.allgather(local)
        out = [None] * len(items)
        for part in allres:
            for i, v in part:
                out[i] = v
        return out
    raise ValueError(backend)


def ensemble(func, seeds, backend="auto", workers=None):
    """Disorder-ensemble average: ``func(seed) -> float``.

    Returns (mean, standard error, all values). Use for any observable that
    has to be averaged over independent energetic-disorder realisations.
    """
    vals = np.asarray(parallel_map(func, list(seeds), backend, workers), float)
    se = vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else np.nan
    return float(vals.mean()), float(se), vals


def split_carriers(n_total, n_parts):
    """Split ``n_total`` walkers into ``n_parts`` nearly equal batches."""
    base, extra = divmod(n_total, n_parts)
    return [base + (1 if i < extra else 0) for i in range(n_parts)]

"""Optional Numba acceleration (falls back to pure Python if unavailable)."""
try:
    from numba import njit, prange  # noqa: F401
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kw):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def deco(f):
            return f
        return deco

    prange = range

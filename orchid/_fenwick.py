"""Fenwick (binary indexed) tree for O(log n) rate selection in KMC.

``tree`` is a float64 array of length n (1-based internally). Rates are
updated with :func:`fw_add` (delta), the total is :func:`fw_total`, and
:func:`fw_find` returns the event index i such that
prefix(i-1) <= u < prefix(i).
"""
import numpy as np
from ._jit import njit


@njit(cache=True)
def fw_build(tree, vals):
    n = vals.shape[0]
    for i in range(n):
        tree[i] = vals[i]
    for i in range(1, n + 1):
        j = i + (i & (-i))
        if j <= n:
            tree[j - 1] += tree[i - 1]


@njit(cache=True)
def fw_add(tree, i, delta):
    n = tree.shape[0]
    i += 1
    while i <= n:
        tree[i - 1] += delta
        i += i & (-i)


@njit(cache=True)
def fw_prefix(tree, i):
    """sum of vals[0..i-1]"""
    s = 0.0
    while i > 0:
        s += tree[i - 1]
        i -= i & (-i)
    return s


@njit(cache=True)
def fw_total(tree):
    return fw_prefix(tree, tree.shape[0])


@njit(cache=True)
def fw_find(tree, u):
    n = tree.shape[0]
    step = 1
    while step * 2 <= n:
        step *= 2
    pos = 0
    rem = u
    while step > 0:
        nxt = pos + step
        if nxt <= n and tree[nxt - 1] <= rem:
            pos = nxt
            rem -= tree[nxt - 1]
        step //= 2
    if pos >= n:
        pos = n - 1
    return pos

"""First-arrival travel times on the 70 x 70, 10 m grid: shortest paths through the grid (Moser 1991).

Every node connects to its neighbours in 32 directions (offsets of up to 3 cells); an edge takes
its length times the mean slowness sampled along it. Dijkstra's algorithm from each source gives
the first arrival everywhere, including diving and head waves; the surface row holds the receivers.
"""
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

N, SPACING = 70, 10.
SOURCES = (0, 17, 34, 52, 69)                 # OpenFWI's source cells at the surface (augment.py)


def stencil():
    """Edges grouped by direction: from-node, to-node, length (m), the cells sampled along each edge."""
    z, x = np.mgrid[:N, :N]
    groups = []
    for dz in range(4):
        for dx in range(-3, 4):
            if (dz == 0 and dx <= 0) or math.gcd(dz, abs(dx)) != 1:
                continue
            ok = (z + dz < N) & (x + dx >= 0) & (x + dx < N)
            t = np.linspace(0, 1, 2 * max(dz, abs(dx)) + 1)
            cells = np.rint(z[ok][:, None] + t * dz).astype(int) * N + np.rint(x[ok][:, None] + t * dx).astype(int)
            groups.append((z[ok] * N + x[ok], (z[ok] + dz) * N + x[ok] + dx, math.hypot(dz, dx) * SPACING, cells))
    return groups


STENCIL = stencil()
FROM, TO = (np.concatenate([group[i] for group in STENCIL]) for i in (0, 1))


def first_arrivals(velocity):
    """Velocity [70,70] (m/s, depth first) -> first-arrival times [5 sources, 70 receivers] in s."""
    slowness = 1 / np.maximum(np.asarray(velocity, np.float64).ravel(), 100.)
    time = np.concatenate([length * slowness[cells].mean(1) for _, _, length, cells in STENCIL])
    graph = csr_matrix((time, (FROM, TO)), shape=(N * N, N * N))
    return dijkstra(graph, directed=False, indices=list(SOURCES))[:, :N]

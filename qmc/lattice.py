"""Periodic (translation-invariant) lattices shared by the QMC and Chebyshev codes.

The site indexing here is the single source of truth: the same object emits the
DSQSS input and the ``Couplings/*.hdf5`` matrix consumed by the Chebyshev code,
so site indices agree by construction and the two outputs can be compared
site by site without solving a graph-isomorphism problem.
"""

import itertools

import numpy as np

# Chebyshev convention: H = sum_{i<j} J_ij S_i.S_j + h_z sum_i S^z_i
# DSQSS convention:     H = sum_<ij> [-Jz S^z S^z - (Jxy/2)(S+S- + h.c.)] - h sum_i S^z
# hence Jz = Jxy = -J and h = -h_z.


class PeriodicLattice:
    """A hypercubic lattice with periodic boundaries and uniform coupling J."""

    def __init__(self, size, J=1.0, name=None):
        self.size = list(size)
        self.dim = len(self.size)
        self.J = float(J)
        self.nsites = int(np.prod(self.size))
        self.name = name or self._default_name()
        self.coords = np.array(
            [self._index_to_coord(i) for i in range(self.nsites)], dtype=np.int64
        )

    def _default_name(self):
        kind = {1: "Chain", 2: "Square", 3: "Cube"}.get(self.dim, f"Hyper{self.dim}D")
        return f"{kind}_NN_PBC_N={self.nsites}"

    def _index_to_coord(self, index):
        """Row-major decomposition matching dsqss.util.index2coord."""
        coord = []
        rest = index
        for length in self.size:
            coord.append(rest % length)
            rest //= length
        return coord

    def coupling_matrix(self):
        """Symmetric J_ij with J on nearest-neighbour bonds, zero elsewhere.

        Contributions are accumulated rather than assigned so that bond
        multiplicity matches DSQSS's lattice generator: along a periodic
        direction of length 2 the +1 and -1 neighbours are the same site and
        DSQSS lays down two bonds, giving an effective 2J there.
        """
        J_ij = np.zeros((self.nsites, self.nsites), dtype=np.float64)
        for i in range(self.nsites):
            for d in range(self.dim):
                if self.size[d] < 2:
                    continue
                neighbour = self.coords[i].copy()
                neighbour[d] = (neighbour[d] + 1) % self.size[d]
                j = self._coord_to_index(neighbour)
                J_ij[i, j] += self.J
                J_ij[j, i] += self.J
        return J_ij

    def _coord_to_index(self, coord):
        index = 0
        stride = 1
        for d in range(self.dim):
            index += int(coord[d]) * stride
            stride *= self.size[d]
        return index

    def displacement_vector(self, i, j):
        """Minimum-image displacement from site i to site j, as dsqss does it."""
        dr = self.coords[j] - self.coords[i]
        for d in range(self.dim):
            if dr[d] <= -self.size[d] // 2:
                dr[d] += self.size[d]
            elif dr[d] > self.size[d] // 2:
                dr[d] -= self.size[d]
        return tuple(int(x) for x in dr)

    @property
    def is_bipartite(self):
        """Hypercubic lattices with periodic boundaries are bipartite iff every
        side length is even (an odd ring closes on a frustrated bond)."""
        return all(length % 2 == 0 for length in self.size if length > 2)

    def marshall_sign(self, i, j):
        """Staggered sign (-1)^(sum of displacement components) between two sites.

        DSQSS makes an antiferromagnet sign-problem-free by taking absolute
        values of the off-diagonal weights (Marshall's rule, algorithm.py:255).
        The worm therefore measures the transverse correlator in the rotated
        frame; multiplying by this sign returns it to the physical frame. The
        diagonal S^z S^z channel is unaffected.
        """
        return -1.0 if sum(self.displacement_vector(i, j)) % 2 else 1.0

    def write_coupling_file(self, path):
        """Write the Chebyshev-format coupling file for this lattice."""
        import h5py

        with h5py.File(path, "w") as f:
            group = f.create_group("all")
            group.create_dataset("J_ij", data=self.coupling_matrix(), dtype="float64")
            group.attrs["num_Spins"] = self.nsites
        return path

    def std_toml(self, beta, ntau, h_z=0.0, mc=None, seed=31415, dispfile="disp.xml",
                 wvfile="wv.xml"):
        """Render the DSQSS ``std.toml`` input for this lattice."""
        mc = mc or {}
        size = self.size[0] if self.dim == 1 else self.size
        lines = [
            "[hamiltonian]",
            'model = "spin"',
            "M = 1",
            f"Jz = {-self.J!r}",
            f"Jxy = {-self.J!r}",
            f"h = {-float(h_z)!r}",
            "",
            "[lattice]",
            'lattice = "hypercubic"',
            f"dim = {self.dim}",
            f"L = {size!r}",
            "bc = true",
            "",
            "[parameter]",
            f"beta = {float(beta)!r}",
            f"ntau = {int(ntau)}",
            f"nset = {mc.get('nset', 10)}",
            f"npre = {mc.get('npre', 1000)}",
            f"ntherm = {mc.get('ntherm', 1000)}",
            f"ndecor = {mc.get('ndecor', 1000)}",
            f"nmcs = {mc.get('nmcs', 20000)}",
            f"seed = {int(seed)}",
            f'dispfile = "{dispfile}"',
            f'wvfile = "{wvfile}"',
            "",
            "[kpoints]",
            "ksteps = 1",
            "",
        ]
        return "\n".join(lines)


def build(spec, J=1.0):
    """Build a lattice from a short spec string such as ``square:4x4`` or ``chain:16``."""
    if ":" not in spec:
        raise ValueError(f"lattice spec must look like 'square:4x4', got {spec!r}")
    kind, dims = spec.split(":", 1)
    size = [int(x) for x in dims.lower().split("x")]
    expected = {"chain": 1, "square": 2, "cube": 3}
    kind = kind.lower()
    if kind not in expected:
        raise ValueError(f"unknown lattice kind {kind!r}; use chain, square or cube")
    if len(size) == 1 and expected[kind] > 1:
        size = size * expected[kind]
    if len(size) != expected[kind]:
        raise ValueError(f"{kind} needs {expected[kind]} dimensions, got {size}")
    return PeriodicLattice(size, J=J)

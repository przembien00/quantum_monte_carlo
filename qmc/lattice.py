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

    def neighbour_shells(self):
        """Sites grouped by distance from site 0, nearest first.

        Entry ``n`` is the n-th neighbour shell: 0 is the site itself, 1 the
        nearest neighbours, 2 the next-nearest, and so on, ordered by squared
        Euclidean distance -- the usual condensed-matter convention, in which
        the square lattice's NNN is the diagonal (1,1) rather than (2,0).

        Each entry is ``(r2, representative_site, displacement, members,
        uniform)``. ``uniform`` is False when the shell contains displacements
        that are *not* related by a symmetry of this lattice, which happens on
        anisotropic lattices: on a 6x4 torus (2,0) and (0,2) are both at r2 = 4
        but are physically distinct, because the two directions have different
        lengths. The measurement always uses the representative's displacement,
        so the result stays well defined; the flag exists so callers can say so.
        """
        groups = {}
        for j in range(self.nsites):
            dr = self.displacement_vector(0, j)
            groups.setdefault(sum(x * x for x in dr), []).append((j, dr))

        def signature(dr):
            # Two displacements can only be symmetry-equivalent if the axes
            # they use have matching lengths, so pair each component with the
            # size of its dimension before comparing.
            return tuple(sorted((abs(x), self.size[d]) for d, x in enumerate(dr)))

        shells = []
        for r2 in sorted(groups):
            members = sorted(groups[r2])
            site, dr = members[0]
            uniform = len({signature(d) for _, d in members}) == 1
            shells.append((r2, site, dr, [m[0] for m in members], uniform))
        return shells

    def shell_sites(self, shells):
        """Translate n-th-neighbour indices into site indices.

        Raises ValueError naming the available range if a shell does not exist,
        which is the common mistake once the lattice is small.
        """
        table = self.neighbour_shells()
        out = []
        for n in shells:
            if not 0 <= n < len(table):
                raise ValueError(
                    f"{self.name} has {len(table)} neighbour shells (0 to "
                    f"{len(table) - 1}); shell {n} does not exist"
                )
            out.append(table[n][1])
        return out

    def write_coupling_file(self, path):
        """Write the Chebyshev-format coupling file for this lattice."""
        import h5py

        with h5py.File(path, "w") as f:
            group = f.create_group("all")
            group.create_dataset("J_ij", data=self.coupling_matrix(), dtype="float64")
            group.attrs["num_Spins"] = self.nsites
        return path

    def _pool_size(self, beta, override=None):
        """Segment/vertex pool size for a run at this beta.

        DSQSS preallocates fixed pools (default 100000) and, on exhaustion,
        prints an error and calls exit(0) -- status *success*, so an overflow
        would otherwise pass as a completed run. The pools must therefore be
        large enough for the whole simulation up front.

        The mean number of off-diagonal vertices is ~ |J| * beta * bonds / 4.
        This uses several times that for headroom against fluctuations, and
        never goes below the DSQSS default. Each element is small, so the memory
        cost is minor next to the worldline itself.

        The bond count is computed analytically -- a d-dimensional torus with
        sides >= 3 has d*N bonds -- rather than from the coupling matrix, which
        is N^2 and cannot be built at the sizes this guards against.
        """
        if override is not None:
            return int(override)
        bonds = self.dim * self.nsites
        mean = abs(self.J) * beta * bonds / 4.0
        return max(100000, int(6 * mean) + 1000)

    def std_toml(self, beta, ntau, h_z=0.0, mc=None, seed=31415,
                 dispfile="disp.xml"):
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
            f"nsegmax = {self._pool_size(beta, mc.get('nsegmax'))}",
            f"nvermax = {self._pool_size(beta, mc.get('nvermax'))}",
            "",
        ]
        # Neither dispfile nor wvfile is declared here, and both are deliberate.
        #
        # wvfile / [kpoints]: C^zz is measured directly in real space on the
        # displacement classes, so the Brillouin-zone machinery is unused --
        # that removes the O(N^2) wavevector file and the O(N^2) structure
        # factor measurement.
        #
        # dispfile: dla_pre would enumerate all N^2 site pairs in Python before
        # writing them out. The caller writes a file holding only the requested
        # classes and points param.in at it afterwards (see run.run_dsqss).
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

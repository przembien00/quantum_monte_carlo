"""Drive a DSQSS/DLA run and convert its output to the Chebyshev schema."""

import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from . import console, correlations, lattice as lattice_mod, parse, storage

INSTALL_DIR = os.environ.get(
    "DSQSS_INSTALL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dsqss_install"),
)


def _binary(name):
    path = os.path.join(INSTALL_DIR, "bin", name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found at {path}. Build DSQSS first (see README) or set "
            "DSQSS_INSTALL to the install prefix."
        )
    return path


_SET_DONE = re.compile(
    r"^\s*(\d+)\s*/\s*(\d+)\s+done\.\s*\[Elapsed:\s*([\d.eE+-]+)\s*sec\."
    r"\s*ETR:\s*([\d.eE+-]+)\s*sec\.\]"
)
_NCYC = re.compile(r"^Determining hyperparameter NCYC\s*:\s*(\d+)")


def _format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 90:
        return f"{minutes:.0f}m{seconds:02.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:.0f}h{minutes:02.0f}m"


def _echo_progress(line, out, style):
    """Translate one line of dla's chatter into a progress line, if it is one.

    Returns True if the line was recognised and printed.
    """
    done = _SET_DONE.match(line)
    if done:
        iset, nset, elapsed, etr = done.groups()
        frac = int(iset) / int(nset)
        filled = int(round(20 * frac))
        bar = style.bar("#" * filled) + "." * (20 - filled)
        print(f"  [{bar}] {style.value(f'set {int(iset):>3}/{nset}')}  "
              f"{style.value(f'{frac*100:3.0f}%')}  "
              f"elapsed {_format_seconds(elapsed):>6}  "
              f"remaining {_format_seconds(etr):>6}",
              file=out, flush=True)
        return True
    ncyc = _NCYC.match(line)
    if ncyc:
        print(f"  calibrated worm cycles per sweep: "
              f"{style.value(ncyc.group(1))}", file=out, flush=True)
        return True
    return False


def _describe_sites(lat, spin_sites, spin_shells):
    """Render the sites line, naming the shell each site stands for."""
    table = {s: (n, dr) for n, (_, s, dr, _, _) in enumerate(lat.neighbour_shells())}
    names = {0: "on-site", 1: "nearest", 2: "next-nearest"}
    parts, mixed = [], []
    for site in spin_sites:
        n, dr = table.get(site, (None, lat.displacement_vector(0, site)))
        label = f"site {site}"
        if n is not None:
            label += f" = shell {n}"
            if n in names:
                label += f" ({names[n]})"
        parts.append(f"{label} {tuple(dr)}")
    if spin_shells is not None:
        for n, (_, _, _, _, uniform) in enumerate(lat.neighbour_shells()):
            if n in spin_shells and not uniform:
                mixed.append(n)
    return parts, mixed


def print_banner(lat, beta, ntau, spin_sites, h_z, mc, seed, ncores,
                 spin_shells=None, out=sys.stdout, style=None):
    """Summarize what is about to be run.

    The destination is deliberately not shown here: the name is only settled
    once the run succeeds, so it is reported at the end instead.
    """
    style = style or console.Style(stream=out)
    mc = mc or {}
    nset = mc.get("nset", 10)
    nmcs = mc.get("nmcs", 20000)
    ntherm = mc.get("ntherm", 1000)
    geometry = "x".join(str(x) for x in lat.size)
    order = "antiferromagnetic" if lat.J > 0 else "ferromagnetic"
    described, mixed = _describe_sites(lat, spin_sites, spin_shells)

    # (label, emphasised value, trailing detail)
    rows = [
        ("model", "isotropic Heisenberg", "S = 1/2"),
        ("lattice", lat.name, f"{geometry}, periodic"),
        ("spins", f"{lat.nsites}", ""),
        ("coupling", f"J = {lat.J:+g}", order),
        ("field", f"h_z = {h_z:g}", ""),
        ("temperature", f"beta = {beta:g}", ""),
        ("imaginary time", f"{ntau} points", f"delta_tau = {beta / ntau:.4g}"),
        ("correlations", described[0], "displacement from site 0"),
    ]
    rows += [("", d, "") for d in described[1:]]
    rows += [
        ("sampling", f"{nset} x {nmcs} sweeps", f"{ntherm} thermalization"),
        ("parallel", f"{ncores} MPI rank" + ("s" if ncores != 1 else ""), ""),
        ("seed", f"{seed}", ""),
    ]
    width = max(len(label) for label, _, _ in rows)
    rule = "=" * 72
    print(rule, file=out)
    print(style.head(" worldline QMC  (DSQSS/DLA)"), file=out)
    print(rule, file=out)
    for label, value, detail in rows:
        tail = f"  {detail}" if detail else ""
        print(f"  {label:<{width}}  {style.value(value)}{tail}", file=out)
    if mixed:
        which = ", ".join(str(n) for n in mixed)
        print(f"  {style.warn('note:')} on this lattice shell {which} contains "
              f"displacements that are not symmetry-equivalent (the sides have "
              f"different lengths); the one shown above is measured.", file=out)
    print(rule, file=out, flush=True)


def run_dsqss(lat, beta, ntau, workdir, spin_sites=(0,), h_z=0.0, mc=None,
              seed=31415, ncores=1, progress=False, out=sys.stdout, style=None):
    """Generate inputs, run dla, and return the paths of its output files.

    With ``progress`` the sampler's per-set output is echoed as it arrives;
    otherwise the run is silent. Either way the full log is retained so that a
    failure can be reported with context.
    """
    style = style or console.Style(stream=out)
    os.makedirs(workdir, exist_ok=True)
    std_path = os.path.join(workdir, "std.toml")
    with open(std_path, "w") as f:
        f.write(lat.std_toml(beta, ntau, h_z=h_z, mc=mc, seed=seed))

    subprocess.run([_binary("dla_pre"), "std.toml"], cwd=workdir, check=True,
                   capture_output=True, text=True)

    # std.toml declares no dispfile, so dla_pre skips its O(N^2) enumeration of
    # every site pair. Write the file covering just the requested displacement
    # classes and point param.in at it.
    dispfile = "disp.xml"
    parse.write_site_displacements(
        os.path.join(workdir, dispfile), lat, list(spin_sites)
    )
    _set_param(os.path.join(workdir, "param.in"), "dispfile", dispfile)

    command = [_binary("dla"), "param.in"]
    if ncores > 1:
        command = ["mpirun", "-n", str(ncores)] + command
    started = time.time()

    # Streamed rather than captured, so progress appears while the sampler runs
    # instead of arriving in one lump at the end.
    log = []
    proc = subprocess.Popen(command, cwd=workdir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log.append(line)
        if progress:
            _echo_progress(line.rstrip("\n"), out, style)
    returncode = proc.wait()
    elapsed = time.time() - started

    if returncode != 0:
        raise RuntimeError("dla failed:\n" + "".join(log)[-2000:])

    if progress:
        print(f"  {style.ok('sampling complete')} in "
              f"{style.value(_format_seconds(elapsed))}", file=out, flush=True)

    return {
        "cf": os.path.join(workdir, "cf.dat"),
        "sf": _find_sf_output(workdir),
        "log": os.path.join(workdir, "sample.log"),
        "disp": os.path.join(workdir, "disp.xml"),
        "seconds": elapsed,
    }


def _set_param(path, key, value):
    """Set ``key = value`` in a dla ``param.in``, appending it if absent."""
    lines = open(path).read().splitlines()
    for index, line in enumerate(lines):
        if line.split("=")[0].strip() == key:
            lines[index] = f"{key} = {value}"
            break
    else:
        lines.append(f"{key} = {value}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _shells_of(lat, spin_sites):
    """Neighbour-shell index of each site, or -1 if it is not a shell head."""
    table = {s: n for n, (_, s, _, _, _) in enumerate(lat.neighbour_shells())}
    return [table.get(s, -1) for s in spin_sites]


def _find_sf_output(workdir):
    """Locate the structure-factor output named in param.in."""
    param = os.path.join(workdir, "param.in")
    if os.path.exists(param):
        for line in open(param):
            if line.startswith("P SFOUTFILE"):
                name = line.split("=", 1)[1].strip()
                candidate = os.path.join(workdir, name)
                if os.path.exists(candidate):
                    return candidate
    fallback = os.path.join(workdir, "sf.dat")
    return fallback if os.path.exists(fallback) else None


def convert(lat, beta, ntau, outputs, spin_sites, h_z=0.0, mc=None, ncores=1,
            seed=31415, project_name="", extension="", spin_shells=None):
    """Assemble the correlation tensor and the parameter block for storage."""
    kinds = parse.read_displacement_kinds(outputs["disp"])
    nkinds = max(kinds.values()) + 1

    cf_results = parse.read_results(outputs["cf"])
    cf_mean, cf_err = parse.collect_tau_series(cf_results, "C", nkinds, ntau)
    cxx, cxx_err = correlations.transverse_from_cf(cf_mean, cf_err)

    # The patched build adds the antisymmetric channel; without it C^xy stays zero.
    af_mean, af_err = parse.collect_tau_series(cf_results, "A", nkinds, ntau)
    has_xy = not np.all(np.isnan(af_mean))
    if has_xy:
        cxy, cxy_err = correlations.xy_from_antisymmetric(af_mean, af_err)

    # C^zz comes from the patched real-space estimator, reported as D entries
    # on the same displacement classes as the transverse channel. The older
    # route (Fourier transform of the structure factor over the full Brillouin
    # zone) gave the same answer but cost O(N^2); see docs/estimators.md.
    czz, czz_err = parse.collect_tau_series(cf_results, "D", nkinds, ntau)
    if np.all(np.isnan(czz)):
        raise RuntimeError(
            "no D entries in the correlation output, so C^zz is unavailable. "
            "The engine build is missing the real-space S^z S^z estimator; "
            "apply patches/dsqss_estimators.patch and rebuild."
        )
    if np.any(np.isnan(czz)):
        raise RuntimeError(
            f"real-space C^zz covers only "
            f"{nkinds - int(np.isnan(czz).any(axis=1).sum())} of {nkinds} "
            "displacement classes; the output looks truncated."
        )

    # Always class C. Both channels are measured on every run, so emitting the
    # collapsed class-A layout at h_z = 0 would discard the independent C^zz
    # estimate, which is the cheapest available check on C^xx (the two must
    # agree by isotropy). Consumers read the layout from `correlation_rows`.
    rows = correlations.CLASS_C_ROWS
    nrows = len(rows)

    corr_re, corr_im = correlations.build_tensor(nrows, ntau, len(spin_sites))
    stds_re, stds_im = correlations.build_tensor(nrows, ntau, len(spin_sites))

    for index, site in enumerate(spin_sites):
        kind = kinds[(0, site)]
        # Undo the Marshall gauge that makes the antiferromagnet sign-free.
        stagger = lat.marshall_sign(0, site) if lat.J > 0 else 1.0
        corr_re[index, 0] = stagger * cxx[kind]  # xx
        stds_re[index, 0] = cxx_err[kind]
        if has_xy:                               # xy and yx = -xy, purely imaginary
            corr_im[index, 1] = stagger * cxy[kind]
            stds_im[index, 1] = cxy_err[kind]
            corr_im[index, 2] = -stagger * cxy[kind]
            stds_im[index, 2] = cxy_err[kind]
        # C^zz is diagonal and so gauge-invariant: no staggering.
        corr_re[index, 3] = czz[kind]
        stds_re[index, 3] = czz_err[kind]

    params = {
        "num_Spins": np.int32(lat.nsites),
        "num_HilbertSpaceDimension": np.int32(2**lat.nsites if lat.nsites < 63 else -1),
        "src_file": f"{lat.name}.hdf5",
        "spin_model": "ISO",
        "rescale": np.float64(1.0),
        "spin_sites": ",".join(str(s) for s in spin_sites),
        # Which neighbour shell each stored site stands for, so the file is
        # readable without reconstructing the lattice geometry.
        "spin_shells": ",".join(
            str(n) for n in (spin_shells if spin_shells is not None
                             else _shells_of(lat, spin_sites))),
        "spin_displacements": ";".join(
            ",".join(str(x) for x in lat.displacement_vector(0, s))
            for s in spin_sites),
        "evol_type": "imaginary",
        "Tmax": np.float64(beta),
        "beta": np.float64(beta),
        "h_z": np.float64(h_z),
        "J": np.float64(lat.J),
        "lattice_size": ",".join(str(x) for x in lat.size),
        "num_TimePoints": np.int32(ntau),
        "delta_t": np.float64(beta / ntau),
        "num_Cores": np.int32(ncores),
        "num_Vectors_Per_Core": np.int32((mc or {}).get("nmcs", 20000)),
        "original project_name": project_name,
        "method": "DSQSS/DLA worldline QMC",
        "correlation_rows": ",".join(rows),
        "seed": np.int32(seed),
    }
    return params, corr_re, corr_im, stds_re, stds_im


def run(lattice_spec, beta, ntau, spin_sites=None, spin_shells=None, h_z=0.0,
        J=1.0, mc=None, seed=31415, ncores=1, workdir=None, data_dir="Data",
        project_name="", extension="", keep_workdir=False, progress=True,
        out=sys.stdout):
    """End-to-end: run QMC on a periodic lattice and write one HDF5 file."""
    lat = lattice_mod.build(lattice_spec, J=J)
    if spin_shells is not None and spin_sites is None:
        spin_sites = lat.shell_sites(spin_shells)
    spin_sites = list(spin_sites) if spin_sites else [0]
    for site in spin_sites:
        if not 0 <= site < lat.nsites:
            raise ValueError(f"site {site} outside lattice of {lat.nsites} sites")

    if J > 0 and not lat.is_bipartite:
        raise ValueError(
            f"{lat.name} is not bipartite (odd side length), so the antiferromagnet "
            "has a sign problem and the Marshall gauge does not apply. Use even side "
            "lengths, or J < 0 for the ferromagnet."
        )

    # Resolved before sampling starts so the banner can name the destination.
    out_dir = os.path.join(data_dir, project_name) if project_name else data_dir
    filename = storage.build_filename(
        "ISO", lat.name, spin_sites, beta, h_z=h_z, extension=extension, J=lat.J
    )
    style = console.Style(stream=out)
    if progress:
        print_banner(lat, beta, ntau, spin_sites, h_z, mc, seed, ncores,
                     spin_shells=spin_shells, out=out, style=style)

    workdir = workdir or os.path.join(
        ".qmc_work", f"{lat.name}__beta={beta}__h={h_z}__seed={seed}"
    )
    outputs = run_dsqss(lat, beta, ntau, workdir, spin_sites=spin_sites,
                        h_z=h_z, mc=mc, seed=seed, ncores=ncores,
                        progress=progress, out=out, style=style)
    params, corr_re, corr_im, stds_re, stds_im = convert(
        lat, beta, ntau, outputs, spin_sites, h_z=h_z, mc=mc, ncores=ncores,
        seed=seed, project_name=project_name, extension=extension,
        spin_shells=spin_shells,
    )

    # Resolved only now: the destination is settled at write time, so the name
    # reported is the one that actually exists on disk.
    dest, n_suffix = storage.resolve_path(os.path.join(out_dir, filename))
    reused = n_suffix == storage.MAX_TRIES - 1 and os.path.exists(dest)
    path = storage.store(
        dest, params, spin_sites, corr_re, corr_im, stds_re, stds_im,
        runtime={"qmc_s": np.float64(outputs["seconds"]),
                 "total_s": np.float64(outputs["seconds"])},
    )
    if progress:
        _warn_nonfinite_errors(stds_re, stds_im, out=out, style=style)
        _print_destination(path, n_suffix, reused, out=out, style=style)
    if not keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return path


def _warn_nonfinite_errors(stds_re, stds_im, out=sys.stdout, style=None):
    """Flag error bars that came back non-finite.

    DSQSS forms the error as sqrt(<x^2> - <x>^2). With few sets and a barely
    fluctuating observable that difference can come out slightly negative
    through cancellation, making the square root NaN. The correlations
    themselves are unaffected, so this is reported rather than treated as fatal.
    """
    style = style or console.Style(stream=out)
    bad = int(np.count_nonzero(~np.isfinite(stds_re))
              + np.count_nonzero(~np.isfinite(stds_im)))
    if bad:
        print(f"  {style.warn('warning:')} {bad} error bar(s) are non-finite "
              f"and stored as NaN; the correlations themselves are unaffected. "
              f"Raise {style.value('--nset')} for a stable variance estimate.",
              file=out)


def _print_destination(path, n_suffix, reused, out=sys.stdout, style=None):
    """Report where the results ended up, once they are safely written."""
    style = style or console.Style(stream=out)
    if reused:
        print(f"  {style.warn('note:')} {storage.MAX_TRIES} name variants "
              f"existed; overwrote the last", file=out)
    elif n_suffix:
        plural = "" if n_suffix == 1 else "s"
        print(f"  {style.warn('note:')} base name taken, appended "
              f"{n_suffix} 'X'{plural}", file=out)
    print(f"  {style.ok('wrote')} {style.path(path)}", file=out, flush=True)

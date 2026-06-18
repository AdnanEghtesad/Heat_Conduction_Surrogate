#!/usr/bin/env python3
"""Dataset generator for the heat-surrogate take-home.

Randomly samples geometry + physics parameters, then for each sample runs the
geometry generator and the steady-heat solver, writing both artifacts into a
dedicated per-sample directory:

    <output_dir>/sample_000/geometry.msh    # tessellated rectangle-with-hole
    <output_dir>/sample_000/solution.vtp    # full temperature field
    <output_dir>/sample_001/...
    ...
    <output_dir>/manifest.csv               # one row per sample (inputs)

Parameters are drawn uniformly from the ranges in the config.

Usage
-----
    python generate_dataset.py                    # uses dataset_config.yaml
    python generate_dataset.py --config other.yaml
"""

import argparse
import contextlib
import csv
import io
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import skfem
import yaml

from geometry_generator import generate_geometry
from simulator import save_solution, solve_heat

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "dataset_config.yaml"
DEFAULT_LOG_FILE = SCRIPT_DIR / "generate_dataset.log"

log = logging.getLogger("generate_dataset")


def setup_logging(log_file: Path) -> None:
    """Log to both stdout and a file, both at DEBUG level."""
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmt)
        log.addHandler(handler)


def sample_parameters(rng: np.random.Generator, ranges: dict) -> dict:
    """Draw one parameter set, sampling each value uniformly from its range."""
    def uniform(name: str) -> float:
        lo, hi = ranges[name]
        return float(rng.uniform(lo, hi))

    return {
        "length": uniform("length"),
        "breadth": uniform("breadth"),
        "radius": uniform("radius"),
        "q": uniform("q"),
        "T_left": uniform("T_left"),
        "T_right": uniform("T_right"),
    }


def generate_sample(params: dict, conductivity: float, element_fraction: float,
                    sample_dir: Path, bounds: dict) -> None:
    """Generate one geometry + solution pair into ``sample_dir``.

    Also writes ``params.json`` recording the exact inputs that produced this
    sample (the surrogate's input vector, plus the fixed conductivity).
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "params.json", "w") as f:
        json.dump({"conductivity": conductivity, **params}, f, indent=2)

    mesh_path = generate_geometry(
        length=params["length"],
        breadth=params["breadth"],
        radius=params["radius"],
        element_fraction=element_fraction,
        output_path=sample_dir / "geometry.msh",
        bounds=bounds,
    )
    mesh = skfem.Mesh.load(str(mesh_path))
    temperature = solve_heat(
        mesh,
        conductivity=conductivity,
        source=params["q"],
        t_left=params["T_left"],
        t_right=params["T_right"],
    )
    save_solution(
        mesh,
        temperature,
        output_path=sample_dir / "solution.vtp",
        conditions={"conductivity": conductivity, **params},
    )


@contextlib.contextmanager
def _quiet():
    """Silence stdout/stderr from the meshing/IO libraries for one sample.

    The logging handlers keep their own references to the real streams, so
    progress logging is unaffected.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Path to the YAML config (default: {DEFAULT_CONFIG.name}).",
    )
    parser.add_argument(
        "--log-file", type=Path, default=DEFAULT_LOG_FILE,
        help=f"Path to the log file (default: {DEFAULT_LOG_FILE.name}).",
    )
    args = parser.parse_args()

    setup_logging(args.log_file)
    cfg = load_config(args.config)

    ranges = cfg["ranges"]
    bounds = {k: ranges[k] for k in ("length", "breadth", "radius")}
    rng = np.random.default_rng(cfg["seed"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    n = cfg["num_samples"]

    log.info("Generating %d samples into %s (seed=%s)", n, output_dir, cfg["seed"])

    rows = []
    for i in range(n):
        params = sample_parameters(rng, ranges)
        sample_dir = output_dir / f"sample_{i:03d}"
        log.info("[%d/%d] %s", i + 1, n, sample_dir.name)
        try:
            with _quiet():
                generate_sample(
                    params, cfg["conductivity"], cfg["element_fraction"], sample_dir, bounds
                )
        except (Exception, SystemExit) as err:
            log.warning("Sample %s skipped (%s)", sample_dir.name, type(err).__name__)
            shutil.rmtree(sample_dir, ignore_errors=True)
            continue
        rows.append({"sample": sample_dir.name, "conductivity": cfg["conductivity"], **params})

    if rows:
        manifest = output_dir / "manifest.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info("Wrote manifest with %d rows to %s", len(rows), manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

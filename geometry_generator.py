#!/usr/bin/env python3
"""Geometry generator for the heat-surrogate take-home.

Generates a 2D rectangle, centred at the origin, with a circular hole in the
middle, and tessellates it into a triangular mesh that is ready to hand
straight to the Poisson (steady-heat) solver.

The rectangle spans:
    x in [-length/2, +length/2]
    y in [-breadth/2, +breadth/2]
with a circular hole of `radius` centred at (0, 0).

The output mesh (gmsh ``.msh``) carries named tags so the solver knows where
to apply boundary conditions:
    - subdomain  "domain" : the 2D area to solve on
    - boundary   "left"   : the x = -length/2 edge
    - boundary   "right"  : the x = +length/2 edge
    - boundary   "top"    : the y = +breadth/2 edge
    - boundary   "bottom" : the y = -breadth/2 edge
    - boundary   "hole"   : the inner circle

Usage
-----
    python geometry_generator.py                     # uses geometry_config.yaml
    python geometry_generator.py --config other.yaml

All parameters live in the YAML config that sits next to this script.
"""

import argparse
import logging
import sys
from pathlib import Path

import gmsh
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "geometry_config.yaml"
DEFAULT_LOG_FILE = SCRIPT_DIR / "geometry_generator.log"

log = logging.getLogger("geometry_generator")


class GeometryError(ValueError):
    """Raised when requested parameters cannot produce a valid geometry."""


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


def validate_parameters(
    length: float, breadth: float, radius: float, bounds: dict
) -> None:
    """Check that each parameter lies within its configured [min, max] bound.

    Raises
    ------
    GeometryError
        If a parameter falls outside its configured [min, max] bound.
    """
    for name, value in (("length", length), ("breadth", breadth), ("radius", radius)):
        lo, hi = bounds[name]
        if not (lo <= value <= hi):
            raise GeometryError(
                f"{name}={value} is outside the allowed range [{lo}, {hi}]."
            )
    log.debug("Parameters checked: length=%s breadth=%s radius=%s", length, breadth, radius)


def generate_geometry(
    length: float,
    breadth: float,
    radius: float,
    element_fraction: float,
    output_path: Path,
    bounds: dict,
) -> Path:
    """Build the tessellated rectangle-with-hole and write it to ``output_path``.

    ``element_fraction`` sets the target triangle edge length *relative* to the
    geometry: ``element_size = element_fraction * min(length, breadth)``. This
    keeps the mesh density (and hence node count) roughly constant across
    differently-sized geometries instead of scaling with area.

    Returns the path to the written ``.msh`` file. The mesh is tagged with the
    "domain" subdomain and the "outer"/"hole" boundaries.
    """
    validate_parameters(length, breadth, radius, bounds)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    element_size = element_fraction * min(length, breadth)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)  # silence gmsh's own stdout
        gmsh.model.add("rectangle_with_hole")
        occ = gmsh.model.occ

        rectangle = occ.addRectangle(-length / 2, -breadth / 2, 0.0, length, breadth)
        hole = occ.addDisk(0.0, 0.0, 0.0, radius, radius)
        cut, _ = occ.cut([(2, rectangle)], [(2, hole)])
        occ.synchronize()

        surfaces = [tag for (dim, tag) in cut if dim == 2]

        # Classify each boundary curve by the location of its centre of mass: a
        # curve sitting on a rectangle wall is tagged with that wall; everything
        # else is the hole boundary.
        edges = {"left": [], "right": [], "top": [], "bottom": [], "hole": []}
        tol = 1e-6 * max(length, breadth)
        for dim, tag in gmsh.model.getBoundary(
            [(2, s) for s in surfaces], oriented=False
        ):
            cx, cy, _ = occ.getCenterOfMass(dim, tag)
            if abs(cx + length / 2) < tol:
                edges["left"].append(tag)
            elif abs(cx - length / 2) < tol:
                edges["right"].append(tag)
            elif abs(cy - breadth / 2) < tol:
                edges["top"].append(tag)
            elif abs(cy + breadth / 2) < tol:
                edges["bottom"].append(tag)
            else:
                edges["hole"].append(tag)

        gmsh.model.addPhysicalGroup(2, surfaces, name="domain")
        for name, tags in edges.items():
            gmsh.model.addPhysicalGroup(1, tags, name=name)

        gmsh.option.setNumber("Mesh.MeshSizeMin", element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.model.mesh.generate(2)

        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        n_tris = len(gmsh.model.mesh.getElements(2)[1][0]) if surfaces else 0

        gmsh.write(str(output_path))
    finally:
        gmsh.finalize()

    log.info(
        "Generated mesh: length=%.4g breadth=%.4g radius=%.4g "
        "element_fraction=%.4g (element_size=%.4g) -> %d nodes, %d triangles",
        length, breadth, radius, element_fraction, element_size, n_nodes, n_tris,
    )
    log.debug("Wrote %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path


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
    log.info("Reading config from %s", args.config)
    cfg = load_config(args.config)

    geom = cfg["geometry"]
    try:
        generate_geometry(
            length=geom["length"],
            breadth=geom["breadth"],
            radius=geom["radius"],
            element_fraction=cfg["mesh"]["element_fraction"],
            output_path=cfg["output"]["path"],
            bounds=cfg["bounds"],
        )
    except GeometryError as err:
        log.error("Geometry generation failed: %s", err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

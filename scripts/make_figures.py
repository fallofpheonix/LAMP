#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from osgeo import gdal, ogr

gdal.UseExceptions()

from lamp.tasks.viewsheds.load_data import load_observers
from lamp.tasks.viewsheds.terrain import world_to_pixel


plt.switch_backend("Agg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate reviewer-facing visibility figures")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "task2"), help="Input data directory")
    p.add_argument("--output-dir", default="outputs", help="Pipeline output directory")
    p.add_argument("--figures-dir", default="figures", help="Figure output directory")
    p.add_argument("--observer-height", type=float, default=1.6, help="Observer eye height in meters")
    return p.parse_args()


def load_raster(path: str) -> tuple[np.ndarray, tuple, str]:
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    return arr, ds.GetGeoTransform(), ds.GetProjection()


def grid_centers(geotransform: tuple, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    origin_x, px_w, _, origin_y, _, px_h = geotransform

    xs = origin_x + (np.arange(cols) + 0.5) * px_w
    ys = origin_y + (np.arange(rows) + 0.5) * px_h

    xx, yy = np.meshgrid(xs, ys)
    return xx, yy


def raster_extent(geotransform: tuple, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    rows, cols = shape
    x0, px_w, _, y0, _, px_h = geotransform
    x1 = x0 + cols * px_w
    y1 = y0 + rows * px_h
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def read_building_rings(buildings_shp: str) -> list[np.ndarray]:
    ds = ogr.Open(buildings_shp)
    if ds is None:
        raise FileNotFoundError(f"Cannot open buildings layer: {buildings_shp}")

    layer = ds.GetLayer(0)
    rings: list[np.ndarray] = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        gname = geom.GetGeometryName().upper()
        polys = []
        if gname == "POLYGON":
            polys = [geom]
        elif gname == "MULTIPOLYGON":
            polys = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]

        for poly in polys:
            ring = poly.GetGeometryRef(0)
            if ring is None:
                continue
            coords = []
            for i in range(ring.GetPointCount()):
                x, y, _ = ring.GetPoint(i)
                coords.append((x, y))
            if len(coords) >= 3:
                rings.append(np.array(coords, dtype=np.float64))

    return rings


def observer_points_xyz(
    marks_shp: str,
    projection_wkt: str,
    geotransform: tuple,
    surface: np.ndarray,
    observer_height: float,
) -> list[dict]:
    observers = load_observers(marks_shp, projection_wkt)
    out = []
    rows, cols = surface.shape

    for obs in observers:
        r, c = world_to_pixel(geotransform, obs["x"], obs["y"])
        if 0 <= r < rows and 0 <= c < cols:
            z = float(surface[r, c] + observer_height)
            out.append({**obs, "row": r, "col": c, "z": z})

    return out


def make_3d_scene(
    fig_path: Path,
    occlusion: np.ndarray,
    visibility_prob: np.ndarray,
    geotransform: tuple,
    building_rings: list[np.ndarray],
    observers_xyz: list[dict],
) -> None:
    xx, yy = grid_centers(geotransform, occlusion.shape)
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdYlBu_r")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    facecolors = cmap(norm(np.clip(visibility_prob, 0.0, 1.0)))
    ax.plot_surface(
        xx,
        yy,
        occlusion,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.95,
    )

    # Building footprint outlines for spatial context.
    z_outline = float(np.nanmax(occlusion) + 0.3)
    for ring in building_rings:
        ax.plot(ring[:, 0], ring[:, 1], zs=z_outline, color="black", linewidth=0.6, alpha=0.5)

    for obs in observers_xyz:
        ax.scatter(obs["x"], obs["y"], obs["z"], c="black", s=30, depthshade=True)
        ax.text(obs["x"], obs["y"], obs["z"] + 0.25, f"obs {obs['id']}", fontsize=8, color="black")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("Visibility Probability")

    ax.set_title("3D Terrain + Buildings + Visibility Gradient")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Elevation (m)")
    ax.view_init(elev=38, azim=-125)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=250)
    plt.close(fig)


def make_2d_overlay(
    fig_path: Path,
    hillshade: np.ndarray,
    visibility_prob: np.ndarray,
    geotransform: tuple,
    building_rings: list[np.ndarray],
    observers_xyz: list[dict],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    extent = raster_extent(geotransform, visibility_prob.shape)

    ax.imshow(hillshade, cmap="gray", origin="upper", extent=extent)
    im = ax.imshow(
        visibility_prob,
        cmap="RdYlBu_r",
        origin="upper",
        extent=extent,
        alpha=0.65,
        vmin=0.0,
        vmax=1.0,
    )

    for ring in building_rings:
        ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=0.6)

    for obs in observers_xyz:
        ax.scatter(obs["x"], obs["y"], c="cyan", edgecolor="black", s=45, zorder=3)
        ax.text(obs["x"] + 0.6, obs["y"] + 0.6, f"obs {obs['id']}", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Visibility Probability")

    ax.set_title("2D Viewshed Probability Overlay")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=250)
    plt.close(fig)


def make_histogram(fig_path: Path, visibility_prob: np.ndarray) -> None:
    vals = visibility_prob[np.isfinite(visibility_prob)]
    vals = vals[(vals >= 0.0) & (vals <= 1.0)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=np.linspace(0, 1, 21), color="#1f77b4", edgecolor="black")
    ax.set_title("Visibility Probability Histogram")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Cell Count")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    occlusion, gt, proj = load_raster(str(output_dir / "occlusion_surface.tif"))
    vis_prob, gt2, _ = load_raster(str(output_dir / "viewshed_probability.tif"))
    hillshade, gt3, _ = load_raster(str(output_dir / "scene_hillshade.tif"))

    if gt != gt2 or gt != gt3:
        raise RuntimeError("Output rasters are not aligned (geotransform mismatch)")

    buildings = read_building_rings(str(data_dir / "BuildingFootprints.shp"))
    observers = observer_points_xyz(
        marks_shp=str(data_dir / "Marks_Brief2.shp"),
        projection_wkt=proj,
        geotransform=gt,
        surface=occlusion,
        observer_height=args.observer_height,
    )

    make_3d_scene(
        fig_path=figures_dir / "3D_visibility_scene.png",
        occlusion=occlusion,
        visibility_prob=vis_prob,
        geotransform=gt,
        building_rings=buildings,
        observers_xyz=observers,
    )
    make_2d_overlay(
        fig_path=figures_dir / "2D_viewshed_overlay.png",
        hillshade=hillshade,
        visibility_prob=vis_prob,
        geotransform=gt,
        building_rings=buildings,
        observers_xyz=observers,
    )
    make_histogram(figures_dir / "visibility_histogram.png", vis_prob)

    print(f"Figures written to: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()

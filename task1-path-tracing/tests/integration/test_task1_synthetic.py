"""
Integration test: run the Task-1 path-tracing pipeline on a small
synthetic DEM + SAR dataset and verify that output artefacts are produced.

Can be imported and called from the CI workflow **or** run directly:

    PYTHONPATH=src python tests/integration/test_task1_synthetic.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _write_synthetic_tif(
    path: Path,
    rows: int = 30,
    cols: int = 30,
    *,
    add_gradient: bool = False,
) -> None:
    """Write a tiny float32 GeoTIFF to *path* with a simple synthetic surface."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    arr = rng.uniform(100.0, 120.0, (rows, cols)).astype(np.float32)
    if add_gradient:
        arr += np.linspace(0, 5, cols, dtype=np.float32)

    transform = from_origin(west=30.0, north=26.0, xsize=1.0, ysize=1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(32636),
        transform=transform,
    ) as dst:
        dst.write(arr, 1)


def _write_synthetic_marks(path: Path, pts: list[tuple[float, float]]) -> None:
    """Write a minimal point shapefile with observer marks."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"id": list(range(len(pts)))},
        geometry=[Point(x, y) for x, y in pts],
        crs="EPSG:32636",
    )
    gdf.to_file(path, driver="ESRI Shapefile")


def _write_synthetic_buildings(path: Path) -> None:
    """Write an empty building footprint shapefile."""
    import geopandas as gpd

    gdf = gpd.GeoDataFrame({"Elevation": []}, geometry=[], crs="EPSG:32636")
    gdf.to_file(path, driver="ESRI Shapefile")


def _load_run_module():
    """Dynamically load the run_pipeline script as a module."""
    spec = importlib.util.spec_from_file_location(
        "run_pipeline",
        ROOT / "scripts" / "run_pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def run_integration() -> None:
    """Execute a minimal end-to-end pipeline run and assert outputs exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        dem_path = base / "dem.tif"
        sar_path = base / "sar.tif"
        marks_path = base / "marks.shp"
        buildings_path = base / "buildings.shp"
        out_dir = base / "out"

        # Synthetic rasters at 30 × 30 pixels (EPSG:32636, 1 m resolution).
        _write_synthetic_tif(dem_path, rows=30, cols=30)
        _write_synthetic_tif(sar_path, rows=30, cols=30, add_gradient=True)

        # Two terminal marks near opposite corners of the 30 m grid.
        _write_synthetic_marks(marks_path, [(30.5, 25.5), (58.5, 6.5)])
        _write_synthetic_buildings(buildings_path)

        from config import PipelineConfig

        cfg = PipelineConfig(
            dem_path=dem_path,
            sar_path=sar_path,
            marks_path=marks_path,
            buildings_path=buildings_path,
            known_paths_path=None,
            path_prior_mode="deterministic",
            out_dir=out_dir,
            samples_per_pair=16,
            max_pairs=1,
            top_k_paths=2,
            noise_temperature=0.10,
        )

        rp_mod = _load_run_module()
        summary = rp_mod.run(cfg)

        assert out_dir.exists(), "Output directory was not created"
        assert (out_dir / "run_summary.json").exists(), "run_summary.json missing"
        assert (out_dir / "predicted_paths.geojson").exists(), "predicted_paths.geojson missing"
        assert (out_dir / "probability_heatmap.tif").exists(), "probability_heatmap.tif missing"
        assert (out_dir / "movement_cost.tif").exists(), "movement_cost.tif missing"
        assert summary["processed_pairs"] >= 1, "No terminal pairs were processed"


if __name__ == "__main__":
    run_integration()
    print("Integration test passed.")

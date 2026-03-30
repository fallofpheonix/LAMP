from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

from lamp.tasks.path_tracing.config import PipelineConfig


ROOT = Path(__file__).resolve().parents[2]


def _write_synthetic_tif(
    path: Path,
    rows: int = 30,
    cols: int = 30,
    *,
    add_gradient: bool = False,
) -> None:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    array = rng.uniform(100.0, 120.0, (rows, cols)).astype(np.float32)
    if add_gradient:
        array += np.linspace(0, 5, cols, dtype=np.float32)

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
        dst.write(array, 1)


def _write_synthetic_marks(path: Path, points: list[tuple[float, float]]) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"id": list(range(len(points)))},
        geometry=[Point(x, y) for x, y in points],
        crs="EPSG:32636",
    )
    gdf.to_file(path, driver="ESRI Shapefile")


def _write_synthetic_buildings(path: Path) -> None:
    import geopandas as gpd

    gdf = gpd.GeoDataFrame({"Elevation": []}, geometry=[], crs="EPSG:32636")
    gdf.to_file(path, driver="ESRI Shapefile")


def _load_run_module():
    spec = importlib.util.spec_from_file_location("run_path_tracing", ROOT / "scripts" / "run_path_tracing.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_task1_pipeline_runs_on_synthetic_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dem_path = base / "dem.tif"
        sar_path = base / "sar.tif"
        marks_path = base / "marks.shp"
        buildings_path = base / "buildings.shp"
        out_dir = base / "out"

        _write_synthetic_tif(dem_path, rows=30, cols=30)
        _write_synthetic_tif(sar_path, rows=30, cols=30, add_gradient=True)
        _write_synthetic_marks(marks_path, [(30.5, 25.5), (58.5, 6.5)])
        _write_synthetic_buildings(buildings_path)

        config = PipelineConfig(
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
            calibrate_weights=False,
        )

        run_module = _load_run_module()
        summary = run_module.run(config)

        assert out_dir.exists()
        assert (out_dir / "run_summary.json").exists()
        assert (out_dir / "predicted_paths.geojson").exists()
        assert (out_dir / "probability_heatmap.tif").exists()
        assert (out_dir / "movement_cost.tif").exists()
        assert summary["processed_pairs"] >= 1

from __future__ import annotations

import importlib.util
import sys
import tempfile
from dataclasses import replace
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
    min_value: float = 100.0,
    max_value: float = 120.0,
) -> None:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    array = rng.uniform(min_value, max_value, (rows, cols)).astype(np.float32)
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


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_visibility_coupling",
        ROOT / "scripts" / "evaluate_visibility_coupling.py",
    )
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


def test_visibility_compare_and_sweep_artifacts_on_synthetic_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dem_path = base / "dem.tif"
        sar_path = base / "sar.tif"
        visibility_path = base / "visibility.tif"
        marks_path = base / "marks.shp"
        buildings_path = base / "buildings.shp"
        compare_out_dir = base / "compare_out"
        sweep_out_dir = base / "sweep_out"

        _write_synthetic_tif(dem_path, rows=30, cols=30)
        _write_synthetic_tif(sar_path, rows=30, cols=30, add_gradient=True)
        _write_synthetic_tif(visibility_path, rows=30, cols=30, min_value=0.0, max_value=1.0)
        _write_synthetic_marks(marks_path, [(30.5, 25.5), (58.5, 6.5)])
        _write_synthetic_buildings(buildings_path)

        config = PipelineConfig(
            dem_path=dem_path,
            sar_path=sar_path,
            marks_path=marks_path,
            buildings_path=buildings_path,
            known_paths_path=None,
            path_prior_mode="deterministic",
            visibility_raster=visibility_path,
            visibility_source="deterministic",
            out_dir=compare_out_dir,
            samples_per_pair=16,
            max_pairs=1,
            top_k_paths=2,
            noise_temperature=0.10,
            cost_w_visibility=0.2,
            calibrate_weights=False,
            compare_visibility_coupling=True,
        )

        run_module = _load_run_module()
        comparison = run_module.run(config)

        assert comparison["comparison_mode"] is True
        assert (compare_out_dir / "comparison_summary.json").exists()
        assert (compare_out_dir / "comparison_density_delta.tif").exists()
        assert (compare_out_dir / "baseline" / "run_summary.json").exists()
        assert (compare_out_dir / "visibility_coupled" / "run_summary.json").exists()

        eval_module = _load_eval_module()
        sweep_summary = eval_module.evaluate_visibility_coupling(
            replace(config, compare_visibility_coupling=False, out_dir=sweep_out_dir),
            visibility_weights=[0.0, 0.2],
            output_dir=sweep_out_dir,
        )

        assert len(sweep_summary["runs"]) == 2
        assert (sweep_out_dir / "visibility_sweep_summary.json").exists()
        assert (sweep_out_dir / "visibility_sweep_table.csv").exists()
        assert (sweep_out_dir / "visibility_sweep_report.md").exists()
        assert (sweep_out_dir / "baseline" / "run_summary.json").exists()
        assert (sweep_out_dir / "w_visibility_0p20" / "run_summary.json").exists()

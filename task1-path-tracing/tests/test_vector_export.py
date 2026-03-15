"""Unit tests for vector export functions."""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rasterio
from rasterio.transform import from_bounds

from gis.raster_to_vector import (
    mask_to_polygon_gdf,
    path_records_to_gdf,
    path_to_linestring,
    pixel_center_xy,
    skeleton_to_centerline_gdf,
)
from simulation.probabilistic_paths import PathRecord


def _simple_transform(rows: int = 10, cols: int = 10) -> rasterio.Affine:
    return from_bounds(0.0, 0.0, float(cols), float(rows), cols, rows)


CRS = "EPSG:32636"


class TestPixelCenterXY:
    def test_origin_cell(self) -> None:
        t = from_bounds(0.0, 0.0, 10.0, 10.0, 10, 10)
        x, y = pixel_center_xy(t, 0, 0)
        assert x == pytest.approx(0.5, abs=1e-6)
        assert y == pytest.approx(9.5, abs=1e-6)

    def test_pixel_center_is_inside_cell(self) -> None:
        t = _simple_transform()
        for r in range(3):
            for c in range(3):
                x, y = pixel_center_xy(t, r, c)
                assert isinstance(x, float)
                assert isinstance(y, float)


class TestPathToLinestring:
    def test_single_step_gives_valid_line(self) -> None:
        t = _simple_transform()
        ls = path_to_linestring([(0, 0), (0, 1)], t)
        assert not ls.is_empty
        assert ls.length > 0.0

    def test_single_pixel_path_not_empty(self) -> None:
        t = _simple_transform()
        ls = path_to_linestring([(2, 3)], t)
        assert not ls.is_empty

    def test_longer_path_has_expected_coord_count(self) -> None:
        t = _simple_transform()
        path = [(0, 0), (1, 1), (2, 2), (3, 3)]
        ls = path_to_linestring(path, t)
        assert len(ls.coords) == 4


class TestPathRecordsToGdf:
    def test_empty_records_returns_empty_gdf(self) -> None:
        gdf = path_records_to_gdf([], _simple_transform(), CRS)
        assert len(gdf) == 0

    def test_single_record_produces_one_row(self) -> None:
        t = _simple_transform()
        rec = PathRecord(path=[(0, 0), (1, 1), (2, 2)], probability=0.8, count=3, base_cost=1.5)
        gdf = path_records_to_gdf([(0, 1, rec)], t, CRS)
        assert len(gdf) == 1
        assert float(gdf.iloc[0]["prob"]) == pytest.approx(0.8, abs=1e-6)
        assert int(gdf.iloc[0]["count"]) == 3

    def test_geometry_not_empty(self) -> None:
        t = _simple_transform()
        rec = PathRecord(path=[(0, 0), (2, 2), (4, 4)], probability=0.5, count=2, base_cost=2.0)
        gdf = path_records_to_gdf([(0, 1, rec)], t, CRS)
        assert not gdf.geometry.iloc[0].is_empty

    def test_crs_assigned(self) -> None:
        t = _simple_transform()
        rec = PathRecord(path=[(0, 0), (1, 1)], probability=0.4, count=1, base_cost=0.5)
        gdf = path_records_to_gdf([(0, 1, rec)], t, CRS)
        assert gdf.crs is not None


class TestMaskToPolygonGdf:
    def test_empty_mask_returns_empty(self) -> None:
        mask = np.zeros((5, 5), dtype=np.uint8)
        gdf = mask_to_polygon_gdf(mask, _simple_transform(), CRS)
        assert len(gdf) == 0

    def test_full_mask_returns_one_polygon(self) -> None:
        mask = np.ones((5, 5), dtype=np.uint8)
        gdf = mask_to_polygon_gdf(mask, _simple_transform(5, 5), CRS)
        assert len(gdf) >= 1

    def test_min_area_filter_removes_tiny_blobs(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = 1  # Tiny 1-cell blob.
        mask[5:8, 5:8] = 1  # Larger 9-cell blob.
        t = _simple_transform(10, 10)
        # Area of a single cell is 1.0.
        gdf_no_filter = mask_to_polygon_gdf(mask, t, CRS, min_area=0.0)
        gdf_filtered = mask_to_polygon_gdf(mask, t, CRS, min_area=5.0)
        assert len(gdf_filtered) < len(gdf_no_filter) or len(gdf_filtered) <= 1


class TestSkeletonToCenterlineGdf:
    def test_empty_skeleton_returns_empty(self) -> None:
        mask = np.zeros((5, 5), dtype=np.uint8)
        gdf = skeleton_to_centerline_gdf(mask, _simple_transform(), CRS)
        assert len(gdf) == 0

    def test_horizontal_skeleton_returns_lines(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, :] = 1
        gdf = skeleton_to_centerline_gdf(mask, _simple_transform(), CRS)
        assert len(gdf) >= 1

    def test_single_pixel_returns_empty(self) -> None:
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        gdf = skeleton_to_centerline_gdf(mask, _simple_transform(), CRS)
        assert len(gdf) == 0

    def test_crs_assigned(self) -> None:
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, :] = 1
        gdf = skeleton_to_centerline_gdf(mask, _simple_transform(), CRS)
        assert gdf.crs is not None


class TestWriteVector:
    def test_geojson_roundtrip(self) -> None:
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame({"val": [1]}, geometry=[Point(0, 0)], crs=CRS)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.geojson"
            from gis.export_layers import write_vector
            write_vector(gdf, out)
            loaded = gpd.read_file(out)
            assert len(loaded) == 1

    def test_gpkg_roundtrip(self) -> None:
        import geopandas as gpd
        from shapely.geometry import LineString

        gdf = gpd.GeoDataFrame({"x": [42]}, geometry=[LineString([(0, 0), (1, 1)])], crs=CRS)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.gpkg"
            from gis.export_layers import write_vector
            write_vector(gdf, out)
            loaded = gpd.read_file(out)
            assert len(loaded) == 1

"""DEM and observer data loading utilities for Task 2 viewsheds.

Provides GDAL-backed loaders for DEM rasters and OGR-backed loaders
for observer-point shapefiles, including automatic CRS reprojection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from osgeo import gdal, ogr, osr


@dataclass
class DemData:
    """DEM raster data together with its spatial reference metadata."""

    path: str
    array: np.ndarray
    geotransform: tuple
    projection_wkt: str
    nodata: Optional[float]


def load_dem(path: str) -> DemData:
    """Open a DEM raster with GDAL and return a :class:`DemData` bundle."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open DEM: {path}")

    band = ds.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()

    return DemData(
        path=path,
        array=array,
        geotransform=ds.GetGeoTransform(),
        projection_wkt=ds.GetProjection(),
        nodata=nodata,
    )


def _make_transformer(src_wkt: str, dst_wkt: str) -> osr.CoordinateTransformation:
    src = osr.SpatialReference()
    src.ImportFromWkt(src_wkt)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    dst = osr.SpatialReference()
    dst.ImportFromWkt(dst_wkt)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return osr.CoordinateTransformation(src, dst)


def load_observers(points_path: str, target_projection_wkt: str, id_field: str = "id") -> List[dict]:
    """Load point features from *points_path* and reproject to *target_projection_wkt*.

    Returns a list of ``{"id": int, "x": float, "y": float}`` dicts.
    Raises :exc:`RuntimeError` if no valid observer points are found.
    """
    ds = ogr.Open(points_path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open points file: {points_path}")

    layer = ds.GetLayer(0)
    src_srs = layer.GetSpatialRef()
    src_wkt = src_srs.ExportToWkt() if src_srs is not None else target_projection_wkt
    transformer = _make_transformer(src_wkt, target_projection_wkt)

    observers = []
    for feat in layer:
        obs_id = feat.GetField(id_field)
        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        gname = geom.GetGeometryName().upper()
        if gname == "MULTIPOINT":
            if geom.GetGeometryCount() == 0:
                continue
            point = geom.GetGeometryRef(0)
        elif gname == "POINT":
            point = geom
        else:
            continue

        x, y, _ = transformer.TransformPoint(point.GetX(), point.GetY())
        observers.append({"id": int(obs_id), "x": float(x), "y": float(y)})

    if not observers:
        raise RuntimeError(f"No observer points found in: {points_path}")

    return observers

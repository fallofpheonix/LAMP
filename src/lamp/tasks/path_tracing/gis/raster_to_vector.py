from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import linemerge, unary_union

from lamp.tasks.path_tracing.simulation.probabilistic_paths import PathRecord


def pixel_center_xy(transform: rasterio.Affine, r: int, c: int) -> tuple[float, float]:
    x, y = rasterio.transform.xy(transform, r, c, offset="center")
    return float(x), float(y)


def path_to_linestring(path: Iterable[tuple[int, int]], transform: rasterio.Affine) -> LineString:
    coords = [pixel_center_xy(transform, r, c) for r, c in path]
    if len(coords) < 2:
        coords = coords * 2
    return LineString(coords)


def path_records_to_gdf(
    records: list[tuple[int, int, PathRecord]],
    transform: rasterio.Affine,
    crs: object,
) -> gpd.GeoDataFrame:
    rows = []
    geoms = []
    for src_idx, dst_idx, rec in records:
        geoms.append(path_to_linestring(rec.path, transform))
        rows.append(
            {
                "src": int(src_idx),
                "dst": int(dst_idx),
                "prob": float(rec.probability),
                "count": int(rec.count),
                "cost": float(rec.base_cost),
                "cells": int(len(rec.path)),
            }
        )
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def mask_to_polygon_gdf(mask: np.ndarray, transform: rasterio.Affine, crs: object, min_area: float = 0.0) -> gpd.GeoDataFrame:
    geoms = []
    for geom, value in features.shapes(mask.astype(np.uint8), transform=transform):
        if int(value) != 1:
            continue
        poly = shape(geom)
        if poly.area < min_area:
            continue
        geoms.append(poly)
    if not geoms:
        return gpd.GeoDataFrame({"score": []}, geometry=[], crs=crs)
    return gpd.GeoDataFrame({"score": [1.0] * len(geoms)}, geometry=geoms, crs=crs)


def skeleton_to_centerline_gdf(mask: np.ndarray, transform: rasterio.Affine, crs: object) -> gpd.GeoDataFrame:
    """
    Convert a 1-pixel skeleton mask into merged centerline LineStrings.
    """
    arr = (mask > 0).astype(np.uint8)
    rows, cols = arr.shape
    segments: list[LineString] = []

    # Forward neighbors only, to avoid duplicate segments.
    neighbors = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] == 0:
                continue
            x0, y0 = pixel_center_xy(transform, r, c)
            for dr, dc in neighbors:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if arr[rr, cc] == 0:
                    continue
                x1, y1 = pixel_center_xy(transform, rr, cc)
                if x0 == x1 and y0 == y1:
                    continue
                segments.append(LineString([(x0, y0), (x1, y1)]))

    if not segments:
        return gpd.GeoDataFrame({"pixels": []}, geometry=[], crs=crs)

    unioned = unary_union(segments)
    if isinstance(unioned, LineString):
        geoms = [unioned]
    elif isinstance(unioned, MultiLineString):
        merged = linemerge(unioned)
        if isinstance(merged, LineString):
            geoms = [merged]
        elif isinstance(merged, MultiLineString):
            geoms = [g for g in merged.geoms if not g.is_empty]
        else:
            geoms = [g for g in unioned.geoms if not g.is_empty]
    else:
        geoms = [g for g in segments if not g.is_empty]

    return gpd.GeoDataFrame({"pixels": [1] * len(geoms)}, geometry=geoms, crs=crs)

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from osgeo import gdal, ogr, osr


def _driver_for_vector(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".shp":
        return "ESRI Shapefile"
    if ext == ".gpkg":
        return "GPKG"
    raise ValueError(f"Unsupported vector extension: {ext}")


def write_raster(
    path: str,
    array: np.ndarray,
    geotransform: tuple,
    projection_wkt: str,
    gdal_dtype: int,
    nodata: Optional[float] = None,
) -> None:
    h, w = array.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, w, h, 1, gdal_dtype, options=["COMPRESS=LZW"])
    if ds is None:
        raise RuntimeError(f"Cannot create raster: {path}")

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection_wkt)

    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(array)
    band.FlushCache()
    ds.FlushCache()


def polygonize_raster(
    raster_path: str,
    vector_path: str,
    layer_name: str = "viewshed",
    field_name: str = "value",
) -> None:
    rds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if rds is None:
        raise FileNotFoundError(f"Cannot open raster: {raster_path}")

    band = rds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(rds.GetProjection())

    driver_name = _driver_for_vector(vector_path)
    vdrv = ogr.GetDriverByName(driver_name)
    if os.path.exists(vector_path):
        vdrv.DeleteDataSource(vector_path)

    vds = vdrv.CreateDataSource(vector_path)
    if vds is None:
        raise RuntimeError(f"Cannot create vector: {vector_path}")

    layer = vds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)
    field = ogr.FieldDefn(field_name, ogr.OFTInteger)
    layer.CreateField(field)
    field_idx = layer.GetLayerDefn().GetFieldIndex(field_name)

    err = gdal.Polygonize(band, None, layer, field_idx, [], callback=None)
    if err != 0:
        raise RuntimeError(f"Polygonize failed for {raster_path}")

    vds.FlushCache()


def write_structured_points_vtk(
    path: str,
    volume_zyx: np.ndarray,
    origin_xyz: tuple[float, float, float],
    spacing_xyz: tuple[float, float, float],
    scalar_name: str = "visibility",
) -> None:
    """
    Write a 3D uint8 volume (z,y,x) as legacy VTK structured points.
    """
    if volume_zyx.ndim != 3:
        raise ValueError("volume_zyx must be 3D")

    nz, ny, nx = volume_zyx.shape
    ox, oy, oz = origin_xyz
    sx, sy, sz = spacing_xyz

    arr = volume_zyx.astype(np.uint8)
    npts = nx * ny * nz

    with open(path, "w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("LAMP Visibility Volume\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {ox:.6f} {oy:.6f} {oz:.6f}\n")
        f.write(f"SPACING {sx:.6f} {sy:.6f} {sz:.6f}\n")
        f.write(f"POINT_DATA {npts}\n")
        f.write(f"SCALARS {scalar_name} unsigned_char 1\n")
        f.write("LOOKUP_TABLE default\n")

        # VTK expects x-fastest order, then y, then z.
        for kz in range(nz):
            for ky in range(ny):
                row = arr[kz, ky, :]
                f.write(" ".join(str(int(v)) for v in row))
                f.write("\n")

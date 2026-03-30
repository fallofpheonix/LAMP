from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from osgeo import gdal, ogr


def rasterize_building_heights(
    building_shp_path: str,
    reference_dem_path: str,
    out_raster_path: Optional[str] = None,
    height_field: str = "Elevation",
) -> Tuple[np.ndarray, tuple, str]:
    """Rasterize building relative heights to DEM-aligned grid."""
    ref = gdal.Open(reference_dem_path, gdal.GA_ReadOnly)
    if ref is None:
        raise FileNotFoundError(f"Cannot open DEM: {reference_dem_path}")

    xsize = ref.RasterXSize
    ysize = ref.RasterYSize
    geotransform = ref.GetGeoTransform()
    projection = ref.GetProjection()

    mem_drv = gdal.GetDriverByName("MEM")
    out_ds = mem_drv.Create("", xsize, ysize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    band = out_ds.GetRasterBand(1)
    band.Fill(0.0)
    band.SetNoDataValue(0.0)

    vec = ogr.Open(building_shp_path)
    if vec is None:
        raise FileNotFoundError(f"Cannot open building layer: {building_shp_path}")
    layer = vec.GetLayer(0)

    err = gdal.RasterizeLayer(
        out_ds,
        [1],
        layer,
        options=[f"ATTRIBUTE={height_field}", "ALL_TOUCHED=TRUE"],
    )
    if err != 0:
        raise RuntimeError("Building rasterization failed")

    arr = band.ReadAsArray().astype(np.float64)

    if out_raster_path:
        gtiff = gdal.GetDriverByName("GTiff")
        copy = gtiff.CreateCopy(out_raster_path, out_ds, strict=0)
        if copy is not None:
            copy.FlushCache()

    return arr, geotransform, projection

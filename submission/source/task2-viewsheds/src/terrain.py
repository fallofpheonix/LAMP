from __future__ import annotations
import rasterio
from .lamp_core.terrain import world_to_pixel as _world_to_pixel
from .lamp_core.terrain import pixel_to_world as _pixel_to_world
from .lamp_core.terrain import inside as _inside
from .lamp_core.terrain import bilinear_sample as _bilinear_sample

def world_to_pixel(geotransform: tuple, x: float, y: float) -> tuple[int, int]:
    transform = rasterio.Affine.from_gdal(*geotransform)
    return _world_to_pixel(transform, x, y)

def pixel_to_world(geotransform: tuple, row: int, col: int) -> tuple[float, float]:
    transform = rasterio.Affine.from_gdal(*geotransform)
    return _pixel_to_world(transform, row, col)

def inside(array, row, col):
    return _inside(array.shape, row, col)

def bilinear_sample(array, row, col):
    return _bilinear_sample(array, row, col)

from __future__ import annotations

from osgeo import gdal


def export_hillshade(dem_path: str, out_path: str) -> None:
    """Create a quick-look shaded relief image for QA/visual inspection."""
    gdal.DEMProcessing(
        out_path,
        dem_path,
        processing="hillshade",
        options=gdal.DEMProcessingOptions(azimuth=315, altitude=45),
    )

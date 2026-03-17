from pathlib import Path
import unittest

from core.models import RasterValidation, VectorValidation
from services.dataset_validation_service import find_crs_mismatches, render_dataset_markdown


class TestDatasetValidationService(unittest.TestCase):
    def test_find_crs_mismatches_returns_only_outliers(self) -> None:
        reference = "EPSG:32638"
        rasters = [
            RasterValidation(Path("dem.tif"), reference, (1.0, 1.0), (10, 10), None, 0.0, (0.0, 0.0, 1.0, 1.0)),
            RasterValidation(Path("sar.tif"), "EPSG:4326", (1.0, 1.0), (10, 10), None, 0.0, (0.0, 0.0, 1.0, 1.0)),
        ]
        vectors = [
            VectorValidation(Path("marks.shp"), reference, 2, 0, 0, (0.0, 0.0, 1.0, 1.0)),
            VectorValidation(Path("buildings.shp"), "EPSG:3857", 2, 0, 0, (0.0, 0.0, 1.0, 1.0)),
        ]

        mismatches = find_crs_mismatches(reference, rasters[1:], vectors)
        self.assertEqual([path.name for path in mismatches], ["sar.tif", "buildings.shp"])

    def test_render_dataset_markdown_includes_sections(self) -> None:
        rasters = [
            RasterValidation(Path("dem.tif"), "EPSG:32638", (1.0, 1.0), (10, 10), None, 0.0, (0, 0, 1, 1)),
        ]
        vectors = [
            VectorValidation(Path("marks.shp"), "EPSG:32638", 2, 0, 0, (0, 0, 1, 1)),
        ]

        text = render_dataset_markdown(rasters, vectors, [])

        self.assertIn("# Dataset Integrity Report", text)
        self.assertIn("## 2. Vector Validation", text)
        self.assertIn("## 3. Raster Validation", text)


if __name__ == "__main__":
    unittest.main()

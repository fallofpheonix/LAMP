import json
import unittest
from pathlib import Path


class TestTask2Artifacts(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.out = self.root / "outputs"

    def test_required_outputs_exist(self):
        required = [
            "viewshed.tif",
            "viewshed.shp",
            "viewshed3d.tif",
            "viewshed3d.shp",
            "viewshed_model_metrics.json",
            "task2_completion.md",
        ]
        missing = [f for f in required if not (self.out / f).exists()]
        self.assertEqual(missing, [], f"Missing outputs: {missing}")

    def test_model_metrics_threshold(self):
        p = self.out / "viewshed_model_metrics.json"
        self.assertTrue(p.exists(), "viewshed_model_metrics.json missing")
        m = json.loads(p.read_text(encoding="utf-8"))
        best = m.get("validation_best_threshold_metrics", {})
        self.assertGreaterEqual(float(best.get("f1", 0.0)), 0.5)


if __name__ == "__main__":
    unittest.main()

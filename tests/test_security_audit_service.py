from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from services.security_audit_service import find_path_traversal_risks, render_security_report


class TestSecurityAuditService(unittest.TestCase):
    def test_find_path_traversal_risks_detects_untrusted_open(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            script = temp_path / "sample.py"
            script.write_text("def handler(arg):\n    return open(arg).read()\n", encoding="utf-8")

            risks = find_path_traversal_risks(temp_path)

        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0].path.name, "sample.py")

    def test_render_security_report_handles_clean_scan(self) -> None:
        report = render_security_report([], has_security_tool=False)
        self.assertIn("Dependency scanner available: no", report)
        self.assertIn("no obvious issues found", report)


if __name__ == "__main__":
    unittest.main()

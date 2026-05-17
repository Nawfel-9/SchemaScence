from __future__ import annotations

import unittest

from PIL import Image

from src.agents.spotter import _normalize_bbox
from src.agents.spotter import _refine_symbols_with_image


class SpotterBBoxNormalizationTests(unittest.TestCase):
    def test_qwen_0_1000_coordinates_convert_to_percent(self) -> None:
        self.assertEqual(_normalize_bbox([275, 248, 445, 712]), [27.5, 24.8, 44.5, 71.2])

    def test_percent_coordinates_stay_percent(self) -> None:
        self.assertEqual(_normalize_bbox([10, 20, 30, 40]), [10.0, 20.0, 30.0, 40.0])

    def test_fractional_coordinates_convert_to_percent(self) -> None:
        self.assertEqual(_normalize_bbox([0.1, 0.2, 0.3, 0.4]), [10.0, 20.0, 30.0, 40.0])

    def test_dict_coordinates_are_supported(self) -> None:
        self.assertEqual(
            _normalize_bbox({"x1": 100, "y1": 200, "x2": 300, "y2": 400}),
            [10.0, 20.0, 30.0, 40.0],
        )

    def test_tiny_boxes_are_rejected(self) -> None:
        self.assertIsNone(_normalize_bbox([10, 10, 10.1, 12]))

    def test_geometry_refines_motor_body_on_simple_fixture(self) -> None:
        image = Image.open("data/diagrams/test.png")
        symbols = [
            {
                "symbol_type": "motor",
                "label": "Motor",
                "bbox": [35.0, 35.0, 65.0, 75.0],
                "confidence": 0.95,
            }
        ]
        refined = _refine_symbols_with_image(image, symbols)
        self.assertEqual(refined[0]["symbol_type"], "motor")
        x1, y1, x2, y2 = refined[0]["bbox"]
        self.assertLessEqual(x1, 39.0)
        self.assertLessEqual(y1, 19.0)
        self.assertGreaterEqual(x2, 82.0)
        self.assertLessEqual(y2, 48.0)


if __name__ == "__main__":
    unittest.main()

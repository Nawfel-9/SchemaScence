from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import src.pipeline_spotting as pipeline


class PipelineSpottingTests(unittest.TestCase):
    def test_spot_all_symbols_uses_cache_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            image = Image.new("RGB", (200, 282), "white")
            image.filename = "sample.png"
            cache_file = cache_dir / "sample_detections.json"
            cache_file.write_text(
                '[{"symbol_type": "motor", "bbox_absolute": [1, 2, 3, 4], "confidence": 0.9}]',
                encoding="utf-8",
            )

            with patch.object(pipeline, "CACHE_DIR", cache_dir), patch.object(pipeline, "spot_symbols") as spotter:
                detections = pipeline.spot_all_symbols(image)

            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0]["symbol_type"], "motor")
            spotter.assert_not_called()

    def test_draw_detections_writes_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "detections.png"
            image = Image.new("RGB", (120, 100), "white")
            detections = [
                {
                    "symbol_type": "motor",
                    "label": "M",
                    "bbox_absolute": [20, 20, 80, 70],
                    "confidence": 0.9,
                }
            ]

            pipeline.draw_detections(image, detections, str(out_path))

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

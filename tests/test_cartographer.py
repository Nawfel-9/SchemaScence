from __future__ import annotations

import unittest

from PIL import Image

from src.agents.cartographer import bbox_tile_to_full, merge_detections, tile_image


class CartographerTests(unittest.TestCase):
    def test_small_image_returns_one_unpadded_tile(self) -> None:
        image = Image.new("RGB", (200, 282), "white")
        tiles = tile_image(image)

        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0]["tile_id"], "t_00_00")
        self.assertEqual((tiles[0]["x1"], tiles[0]["y1"], tiles[0]["x2"], tiles[0]["y2"]), (0, 0, 200, 282))
        self.assertEqual(tiles[0]["image"].size, (200, 282))

    def test_large_image_tiles_reach_edges(self) -> None:
        image = Image.new("RGB", (2000, 1500), "white")
        tiles = tile_image(image, tile_size=768, overlap=0.2)

        self.assertEqual(len(tiles), 12)
        self.assertEqual((tiles[-1]["x2"], tiles[-1]["y2"]), (2000, 1500))
        self.assertTrue(all(tile["image"].size == (768, 768) for tile in tiles))

    def test_bbox_tile_to_full_uses_tile_offset(self) -> None:
        image = Image.new("RGB", (2000, 1500), "white")
        tile = tile_image(image, tile_size=768, overlap=0.2)[-1]

        self.assertEqual(bbox_tile_to_full([0, 0, 100, 100], tile, image), [1232, 732, 2000, 1500])

    def test_merge_detections_keeps_highest_confidence_duplicate(self) -> None:
        image = Image.new("RGB", (1000, 1000), "white")
        tile = tile_image(image)[0]
        merged = merge_detections(
            [
                (tile, [{"symbol_type": "motor", "bbox": [10, 10, 20, 20], "confidence": 0.4}]),
                (tile, [{"symbol_type": "motor", "bbox": [11, 11, 21, 21], "confidence": 0.9}]),
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["confidence"], 0.9)
        self.assertEqual(merged[0]["bbox_percent"], [11.0, 11.0, 21.0, 21.0])


if __name__ == "__main__":
    unittest.main()

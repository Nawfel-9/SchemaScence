from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from src.agents.connector import build_graph, graph_to_dict, graph_to_text


class ConnectorTests(unittest.TestCase):
    def test_build_graph_adds_vlm_reported_edge(self) -> None:
        image = Image.new("RGB", (300, 200), "white")
        detections = [
            {"symbol_type": "motor", "label": "M", "bbox_absolute": [50, 50, 100, 100], "confidence": 0.9},
            {"symbol_type": "gate_valve", "label": "V-1", "bbox_absolute": [110, 60, 140, 90], "confidence": 0.8},
        ]

        with patch(
            "src.agents.connector.ask_vlm_json",
            side_effect=[
                {"connected_to": [1], "connection_types": ["pipe"]},
                {"connected_to": [], "connection_types": []},
            ],
        ):
            graph = build_graph(image, detections)

        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(graph.edges[0, 1]["connection_type"], "pipe")

    def test_graph_serializers_are_json_friendly(self) -> None:
        image = Image.new("RGB", (300, 200), "white")
        detections = [
            {"symbol_type": "motor", "label": "M", "bbox_absolute": [50, 50, 100, 100], "confidence": 0.9},
            {"symbol_type": "gate_valve", "label": "V-1", "bbox_absolute": [110, 60, 140, 90], "confidence": 0.8},
        ]

        with patch(
            "src.agents.connector.ask_vlm_json",
            side_effect=[
                {"connected_to": [1], "connection_types": ["pipe"]},
                {"connected_to": [], "connection_types": []},
            ],
        ):
            graph = build_graph(image, detections)

        data = graph_to_dict(graph)
        text = graph_to_text(graph)

        self.assertEqual(data["nodes"][0]["id"], "0")
        self.assertEqual(data["edges"][0]["source"], "0")
        self.assertIn("Node 0 [motor] label=M", text)
        self.assertIn("connected to Node 1 [gate_valve]", text)


if __name__ == "__main__":
    unittest.main()

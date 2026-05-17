from __future__ import annotations

import unittest
from unittest.mock import patch

import networkx as nx
from PIL import Image

from src.agents.reasoner import answer_question


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, symbol_type="centrifugal_pump", label="P001", bbox=[40, 40, 80, 80])
    graph.add_node(1, symbol_type="gate_valve", label="V001", bbox=[100, 40, 130, 70])
    graph.add_edge(0, 1, connection_type="pipe")
    return graph


class ReasonerTests(unittest.TestCase):
    def test_answer_question_returns_graph_only_answer(self) -> None:
        with patch(
            "src.agents.reasoner.ask_vlm_json",
            return_value={
                "action": "answer",
                "answer": "P001 connects to V001",
                "confidence": 0.9,
                "reasoning": "The graph has an edge from P001 to V001.",
            },
        ):
            result = answer_question("What is connected to P001?", _graph(), Image.new("RGB", (200, 150), "white"))

        self.assertEqual(result["answer"], "P001 connects to V001")
        self.assertFalse(result["used_image_lookup"])
        self.assertEqual(result["n_reflections"], 0)
        self.assertEqual(result["confidence"], 0.9)

    def test_answer_question_uses_visual_lookup_when_requested(self) -> None:
        with patch(
            "src.agents.reasoner.ask_vlm_json",
            side_effect=[
                {
                    "action": "lookup",
                    "lookup_target": "P001",
                    "reasoning": "The graph lacks the visible specification.",
                },
                {
                    "action": "answer",
                    "answer": "6 barg",
                    "confidence": 0.8,
                    "reasoning": "The crop shows the pressure action text.",
                },
            ],
        ):
            result = answer_question("What pressure action is shown?", _graph(), Image.new("RGB", (200, 150), "white"))

        self.assertEqual(result["answer"], "6 barg")
        self.assertTrue(result["used_image_lookup"])
        self.assertEqual(result["n_reflections"], 1)
        self.assertEqual(result["confidence"], 0.8)

    def test_answer_question_fails_safely(self) -> None:
        with patch("src.agents.reasoner.ask_vlm_json", side_effect=RuntimeError("model offline")):
            result = answer_question("Anything?", _graph(), Image.new("RGB", (200, 150), "white"))

        self.assertEqual(result["answer"], "unknown")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("failed safely", result["reasoning"])


if __name__ == "__main__":
    unittest.main()

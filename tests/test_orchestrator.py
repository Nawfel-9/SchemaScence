from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
from PIL import Image

import src.orchestrator as orchestrator


class OrchestratorTests(unittest.TestCase):
    def test_schemasense_runs_and_reuses_md5_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "diagram.png"
            Image.new("RGB", (120, 100), "white").save(image_path)
            cache_dir = root / "cache"

            graph = nx.Graph()
            graph.add_node(0, symbol_type="motor", label="M", bbox=[10, 10, 50, 50])
            detections = [{"symbol_type": "motor", "label": "M", "bbox_absolute": [10, 10, 50, 50], "confidence": 0.9}]
            answer = {
                "answer": "motor",
                "reasoning": "Graph contains one motor.",
                "confidence": 0.9,
                "used_image_lookup": False,
            }

            with (
                patch.object(orchestrator, "CACHE_DIR", cache_dir),
                patch.object(orchestrator, "spot_all_symbols", return_value=detections) as spotter,
                patch.object(orchestrator, "build_graph", return_value=graph) as connector,
                patch.object(orchestrator, "answer_question", return_value=answer),
            ):
                first = orchestrator.schemasense(str(image_path), "What is shown?", use_cache=True, use_hybrid=False)
                second = orchestrator.schemasense(str(image_path), "What is shown?", use_cache=True, use_hybrid=False)

            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second["timing"]["spotting_seconds"], 0)
            self.assertEqual(second["timing"]["graph_seconds"], 0)
            self.assertEqual(second["answer"], "motor")
            self.assertEqual(second["graph_stats"], {"nodes": 1, "edges": 0})
            spotter.assert_called_once_with(unittest.mock.ANY, use_cache=False)
            connector.assert_called_once()

    def test_batch_run_augments_questions(self) -> None:
        with patch.object(
            orchestrator,
            "schemasense",
            return_value={
                "answer": "2",
                "timing": {"total_seconds": 0.1},
                "confidence": 0.8,
                "used_image_lookup": False,
                "graph_stats": {"nodes": 1, "edges": 0},
                "cache_hit": True,
            },
        ):
            rows = orchestrator.batch_run([{"diagram": "x.png", "question": "How many?"}])

        self.assertEqual(rows[0]["predicted_answer"], "2")
        self.assertEqual(rows[0]["timing"]["total_seconds"], 0.1)
        self.assertTrue(rows[0]["cache_hit"])

    def test_hybrid_answer_prefers_full_image_verifier(self) -> None:
        graph = nx.Graph()
        graph.add_node(0, symbol_type="controller", label="A", bbox=[10, 10, 50, 50])
        graph_answer = {
            "answer": "wrong graph answer",
            "reasoning": "Sparse graph guess.",
            "confidence": 0.9,
            "used_image_lookup": False,
        }

        with patch(
            "src.baseline.baseline_answer",
            return_value={"answer": "correct visual answer", "confidence": 0.67, "elapsed_seconds": 0.1},
        ):
            final, visual, decision = orchestrator._hybrid_answer(
                image_file=Path("diagram.png"),
                question="What label is shown?",
                question_type="identification",
                graph=graph,
                graph_answer=graph_answer,
                detections=[{"confidence": 0.9}],
                graph_weak=False,
            )

        self.assertEqual(final["answer"], "correct visual answer")
        self.assertEqual(visual["answer"], "correct visual answer")
        self.assertEqual(visual["confidence"], 0.67)
        self.assertEqual(decision["final_answer_source"], "visual_fallback")


if __name__ == "__main__":
    unittest.main()

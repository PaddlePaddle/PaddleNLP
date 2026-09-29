import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


def _load_evaluator():
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *_args, **_kwargs: None
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable: iterable
    requests = types.ModuleType("requests")

    module_path = Path(__file__).with_name("evaluate_from_api.py")
    spec = importlib.util.spec_from_file_location("evaluate_from_api", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"datasets": datasets, "tqdm": tqdm, "requests": requests},
    ):
        spec.loader.exec_module(module)
    return module


class TestUpdateResult(TestCase):
    def test_unparseable_predictions_are_counted_as_wrong(self):
        evaluator = _load_evaluator()
        rows = [
            {
                "category": "math",
                "pred": None,
                "answer": "A",
                "answer_index": 3,
                "options": ["A", "B", "C", "D"],
            },
            {
                "category": "math",
                "pred": "",
                "answer": "B",
                "answer_index": 0,
                "options": ["A", "B", "C", "D"],
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "math_result.json"
            result_path.write_text(json.dumps(rows), encoding="utf-8")

            _, category_record = evaluator.update_result(str(result_path))

        self.assertEqual(
            category_record["math"],
            {"corr": 0.0, "wrong": 2.0},
        )

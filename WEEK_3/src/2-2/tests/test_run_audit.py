import csv
import tempfile
import unittest
from pathlib import Path

import run_audit


class TestJsonExtraction(unittest.TestCase):
    def test_extract_plain_json(self):
        s = '{"incorrect": 1, "incomplete": 0, "inconsistent": 1}'
        got = run_audit.extract_json_dict(s)
        self.assertEqual(got, {"incorrect": 1, "incomplete": 0, "inconsistent": 1})

    def test_extract_json_with_noise(self):
        s = "Here is result:\n```json\n{\"incorrect\":1,\"incomplete\":1,\"inconsistent\":0}\n```"
        got = run_audit.extract_json_dict(s)
        self.assertEqual(got, {"incorrect": 1, "incomplete": 1, "inconsistent": 0})

    def test_extract_fallback_default(self):
        got = run_audit.extract_json_dict("not a json")
        self.assertEqual(got, {"incorrect": 0, "incomplete": 0, "inconsistent": 0})


class TestPromptTemplate(unittest.TestCase):
    def test_prompt_includes_three_label_few_shot_conflict_examples(self):
        prompt = run_audit.build_user_prompt("DS_EXAMPLE", "PP_EXAMPLE")
        self.assertIn("[incorrect example -> 1]", prompt)
        self.assertIn("[incomplete example -> 1]", prompt)
        self.assertIn("[inconsistent example -> 1]", prompt)
        self.assertIn("Output format strictly as JSON", prompt)
        self.assertIn("DS_EXAMPLE", prompt)
        self.assertIn("PP_EXAMPLE", prompt)


class TestPipelineWithMock(unittest.TestCase):
    def test_mock_audit_postprocess(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            in_csv = td_path / "input.csv"
            raw_csv = td_path / "raw.csv"
            processed_csv = td_path / "processed.csv"

            with in_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "app_id",
                        "data_safety_content",
                        "privacy_policy_content",
                        "result",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "app_id": "1",
                        "data_safety_content": "{'data_shared': [], 'data_collected': []}",
                        "privacy_policy_content": "Data Share: some third-party ads.",
                        "result": "",
                    }
                )

            provider = run_audit.MockProvider()
            logger = run_audit.RuntimeLogger(None)
            run_audit.run_audit(in_csv, raw_csv, provider, limit=None, logger=logger, log_every=1)
            run_audit.postprocess_results(raw_csv, processed_csv, logger=logger)
            logger.close()

            with processed_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["incorrect"], "1")
            self.assertEqual(rows[0]["incomplete"], "1")
            self.assertEqual(rows[0]["inconsistent"], "0")


if __name__ == "__main__":
    unittest.main()

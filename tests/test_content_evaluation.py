import json
import unittest
from pathlib import Path

from functions.content_evaluation import (
    ContentEvalDatasetError,
    grounding_contract_violations,
    identity_contract_violations,
    ndcg_at_k,
    precision_at_k,
    summary_pairwise_accuracy,
    title_recall,
    validate_dataset,
)


def record(
    paper_id: str,
    label: str,
    preferred_rank: int | None,
) -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "link": f"https://arxiv.org/abs/{paper_id}",
        "abstract": "A stable evaluation fixture.",
        "label": label,
        "preferred_rank": preferred_rank,
        "primary_bucket": "loop_engineering",
    }


class ContentEvaluationTests(unittest.TestCase):
    def test_unlabeled_import_template_is_valid_before_population(self):
        template_path = Path(__file__).with_name("content_eval_dataset_template.json")
        template = json.loads(template_path.read_text())

        validate_dataset(
            template,
            allow_unlabeled=True,
            require_target_size=False,
        )

    def test_dataset_schema_supports_target_size_without_inventing_labels(self):
        papers = [
            record(
                f"2608.{index:05d}",
                "must_include" if index < 10 else "acceptable",
                index + 1,
            )
            for index in range(60)
        ]
        dataset = {"papers": papers, "summary_pairs": []}

        validate_dataset(dataset)

    def test_dataset_rejects_unlabeled_records_in_scored_mode(self):
        papers = [
            record(f"2608.{index:05d}", "acceptable", None) for index in range(60)
        ]
        papers[0]["label"] = None

        with self.assertRaises(ContentEvalDatasetError):
            validate_dataset({"papers": papers, "summary_pairs": []})

    def test_ranking_metrics_are_deterministic(self):
        papers = [
            record("a", "must_include", 1),
            record("b", "acceptable", 2),
            record("c", "exclude", None),
        ]

        self.assertEqual(title_recall(papers, ["a", "b"]), 1.0)
        self.assertEqual(precision_at_k(papers, ["a", "c"], k=2), 0.5)
        self.assertEqual(ndcg_at_k(papers, ["a", "b", "c"], k=3), 1.0)

    def test_summary_pairwise_preference_metric(self):
        pairs = [
            {"pair_id": "p1", "preferred": "a"},
            {"pair_id": "p2", "preferred": "tie"},
        ]
        score = summary_pairwise_accuracy(pairs, {"p1": "a", "p2": "b"})
        self.assertEqual(score, 0.5)

    def test_identity_and_grounding_contracts(self):
        source = [record("2608.12345", "must_include", 1)]
        generated = [
            {
                "paper_id": "2608.12345",
                "title": source[0]["title"],
                "link": source[0]["link"],
                "grounding_source": "abstract",
                "summary": "A grounded summary.",
            }
        ]

        self.assertEqual(identity_contract_violations(source, generated), [])
        self.assertEqual(grounding_contract_violations(generated), [])


if __name__ == "__main__":
    unittest.main()

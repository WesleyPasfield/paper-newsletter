"""Deterministic evaluation contracts for the newsletter content pipeline."""

import math
from collections.abc import Iterable, Mapping
from typing import Any

LABELS = {"must_include", "acceptable", "exclude"}
PAIRWISE_PREFERENCES = {"a", "b", "tie"}
TARGET_DATASET_RANGE = range(60, 101)


class ContentEvalDatasetError(ValueError):
    """The evaluation dataset does not satisfy its declared schema."""


def validate_dataset(
    dataset: Mapping[str, Any],
    *,
    allow_unlabeled: bool = False,
    require_target_size: bool = True,
) -> None:
    papers = dataset.get("papers")
    summary_pairs = dataset.get("summary_pairs")
    if not isinstance(papers, list) or not isinstance(summary_pairs, list):
        raise ContentEvalDatasetError("papers and summary_pairs must be lists")
    if require_target_size and len(papers) not in TARGET_DATASET_RANGE:
        raise ContentEvalDatasetError("the relevance set must contain 60 to 100 papers")

    seen_ids: set[str] = set()
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            raise ContentEvalDatasetError(f"papers[{index}] must be an object")
        for field in ("paper_id", "title", "link", "abstract"):
            if not isinstance(paper.get(field), str) or not paper[field].strip():
                raise ContentEvalDatasetError(
                    f"papers[{index}].{field} must be a non-empty string"
                )
        paper_id = paper["paper_id"]
        if paper_id in seen_ids:
            raise ContentEvalDatasetError(f"duplicate paper_id: {paper_id}")
        seen_ids.add(paper_id)
        _validate_optional_label(
            paper.get("label"), LABELS, allow_unlabeled, f"papers[{index}].label"
        )
        preferred_rank = paper.get("preferred_rank")
        if preferred_rank is not None and (
            isinstance(preferred_rank, bool)
            or not isinstance(preferred_rank, int)
            or preferred_rank < 1
        ):
            raise ContentEvalDatasetError(
                f"papers[{index}].preferred_rank must be a positive integer or null"
            )
        primary_bucket = paper.get("primary_bucket")
        if primary_bucket is not None and not isinstance(primary_bucket, str):
            raise ContentEvalDatasetError(
                f"papers[{index}].primary_bucket must be a string or null"
            )
        if not allow_unlabeled and not primary_bucket:
            raise ContentEvalDatasetError(
                f"papers[{index}].primary_bucket requires a human label"
            )

    seen_pair_ids: set[str] = set()
    for index, pair in enumerate(summary_pairs):
        if not isinstance(pair, dict):
            raise ContentEvalDatasetError(f"summary_pairs[{index}] must be an object")
        for field in ("pair_id", "paper_id", "summary_a", "summary_b"):
            if not isinstance(pair.get(field), str) or not pair[field].strip():
                raise ContentEvalDatasetError(
                    f"summary_pairs[{index}].{field} must be a non-empty string"
                )
        if pair["pair_id"] in seen_pair_ids:
            raise ContentEvalDatasetError(f"duplicate pair_id: {pair['pair_id']}")
        seen_pair_ids.add(pair["pair_id"])
        if pair["paper_id"] not in seen_ids:
            raise ContentEvalDatasetError(
                f"summary_pairs[{index}].paper_id is absent from papers"
            )
        _validate_optional_label(
            pair.get("preferred"),
            PAIRWISE_PREFERENCES,
            allow_unlabeled,
            f"summary_pairs[{index}].preferred",
        )


def _validate_optional_label(
    value: Any,
    allowed: set[str],
    allow_unlabeled: bool,
    field_name: str,
) -> None:
    if value is None and allow_unlabeled:
        return
    if value not in allowed:
        raise ContentEvalDatasetError(f"{field_name} must be one of {sorted(allowed)}")


def title_recall(
    papers: Iterable[Mapping[str, Any]], retained_ids: Iterable[str]
) -> float:
    retained = set(retained_ids)
    required = {
        paper["paper_id"] for paper in papers if paper.get("label") == "must_include"
    }
    return len(required & retained) / len(required) if required else 1.0


def precision_at_k(
    papers: Iterable[Mapping[str, Any]], ranked_ids: list[str], k: int = 5
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    labels = {paper["paper_id"]: paper.get("label") for paper in papers}
    selected = ranked_ids[:k]
    if not selected:
        return 0.0
    relevant = sum(
        labels.get(paper_id) in {"must_include", "acceptable"} for paper_id in selected
    )
    return relevant / k


def ndcg_at_k(
    papers: Iterable[Mapping[str, Any]], ranked_ids: list[str], k: int = 10
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    records = {paper["paper_id"]: paper for paper in papers}
    gains = [_relevance_gain(records.get(paper_id)) for paper_id in ranked_ids[:k]]
    ideal = sorted(
        (_relevance_gain(paper) for paper in records.values()), reverse=True
    )[:k]
    ideal_dcg = _discounted_gain(ideal)
    return _discounted_gain(gains) / ideal_dcg if ideal_dcg else 1.0


def _relevance_gain(paper: Mapping[str, Any] | None) -> float:
    if not paper:
        return 0.0
    base = {"must_include": 3.0, "acceptable": 1.0, "exclude": 0.0}.get(
        paper.get("label"), 0.0
    )
    preferred_rank = paper.get("preferred_rank")
    if base and isinstance(preferred_rank, int) and preferred_rank > 0:
        return base + 1.0 / preferred_rank
    return base


def _discounted_gain(gains: Iterable[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def summary_pairwise_accuracy(
    summary_pairs: Iterable[Mapping[str, Any]],
    predicted_preferences: Mapping[str, str],
) -> float:
    labeled = [
        pair for pair in summary_pairs if pair.get("preferred") in PAIRWISE_PREFERENCES
    ]
    if not labeled:
        return 1.0
    correct = sum(
        predicted_preferences.get(pair["pair_id"]) == pair["preferred"]
        for pair in labeled
    )
    return correct / len(labeled)


def identity_contract_violations(
    source_papers: Iterable[Mapping[str, Any]],
    generated_papers: Iterable[Mapping[str, Any]],
) -> list[str]:
    source = {paper["paper_id"]: paper for paper in source_papers}
    violations: list[str] = []
    seen: set[str] = set()
    for generated in generated_papers:
        paper_id = generated.get("paper_id")
        if paper_id in seen:
            violations.append(f"duplicate generated paper_id: {paper_id}")
            continue
        seen.add(paper_id)
        expected = source.get(paper_id)
        if expected is None:
            violations.append(f"unknown generated paper_id: {paper_id}")
            continue
        for field in ("title", "link"):
            if generated.get(field) != expected.get(field):
                violations.append(f"{paper_id}: generated {field} changed")
    return violations


def grounding_contract_violations(
    featured_papers: Iterable[Mapping[str, Any]],
) -> list[str]:
    violations: list[str] = []
    for paper in featured_papers:
        paper_id = paper.get("paper_id", "<missing>")
        if paper.get("grounding_source") not in {"abstract", "full_text"}:
            violations.append(f"{paper_id}: invalid grounding_source")
        if not isinstance(paper.get("summary"), str) or not paper["summary"].strip():
            violations.append(f"{paper_id}: missing grounded summary")
    return violations

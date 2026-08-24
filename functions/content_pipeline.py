"""Pure data contracts and transformations for the newsletter content pipeline."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup

TITLE_SCREEN_FLOOR = 0.35
FINAL_RELEVANCE_FLOOR = 0.60
DEFAULT_CANDIDATE_POOL_SIZE = 80
DEFAULT_ABSTRACT_POOL_SIZE = 30
DEFAULT_MAX_FEATURED = 7
DEFAULT_MAX_ADDITIONAL = 3
DEFAULT_MAX_PAPER_AGE_DAYS = 30
DEFAULT_MIN_FEATURED = 3
SELECTION_WINDOWS_DAYS = (7, 14, 30)

ABSTRACT_SCORE_WEIGHTS = {
    "direct_fit": 0.40,
    "evidence_quality": 0.25,
    "production_transferability": 0.20,
    "domain_expert_relevance": 0.10,
    "novelty": 0.05,
}


class ModelOutputError(ValueError):
    """The model returned output that violates a deterministic contract."""


@dataclass
class AbstractEvaluation:
    primary_bucket: str
    direct_fit: float
    evidence_quality: float
    production_transferability: float
    domain_expert_relevance: float
    novelty: float
    uncertainty: float = 0.0
    exclusion_reason: str = ""

    @property
    def total(self) -> float:
        return sum(
            getattr(self, component) * weight
            for component, weight in ABSTRACT_SCORE_WEIGHTS.items()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_bucket": self.primary_bucket,
            "direct_fit": self.direct_fit,
            "evidence_quality": self.evidence_quality,
            "production_transferability": self.production_transferability,
            "domain_expert_relevance": self.domain_expert_relevance,
            "novelty": self.novelty,
            "uncertainty": self.uncertainty,
            "exclusion_reason": self.exclusion_reason,
            "total": self.total,
        }


@dataclass
class Paper:
    title: str
    link: str
    abstract: str = ""
    full_text: str = ""
    interest_score: float = 0.0
    summary: str = ""
    paper_id: str = ""
    authors: list[str] = field(default_factory=list)
    published_date: str = ""
    primary_bucket: str = ""
    grounding_source: str = ""
    component_scores: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.title, self.link))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Paper):
            return False
        return self.title == other.title and self.link == other.link


@dataclass(frozen=True)
class WeeklySelection:
    featured: list[Paper]
    additional: list[Paper]
    selection_mode: str
    source_window_days: int
    valid_candidates: int

    @property
    def selected(self) -> int:
        return len(self.featured)


def canonical_arxiv_id(value: str) -> str:
    """Return a version-independent arXiv identifier, or an empty string."""
    if not value:
        return ""
    match = re.search(
        r"(?:arxiv:\s*|/abs/|/pdf/)?((?:[a-z-]+(?:\.[a-z-]+)?/\d{7}|\d{4}\.\d{4,5}))(?:v\d+)?(?:\.pdf)?(?:[?#].*)?$",
        value.strip(),
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else ""


def canonical_paper_key(paper: Paper) -> str:
    arxiv_id = paper.paper_id or canonical_arxiv_id(paper.link)
    if arxiv_id and not arxiv_id.startswith(("url:", "title:")):
        return f"arxiv:{arxiv_id}"
    parsed = urlsplit(paper.link.strip())
    normalized_url = urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")
    )
    if normalized_url:
        return f"url:{normalized_url}"
    return f"title:{' '.join(paper.title.casefold().split())}"


def clamp_score(value: Any, field_name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(f"{field_name} must be numeric") from exc
    if not 0.0 <= score <= 1.0:
        raise ModelOutputError(f"{field_name} must be between 0 and 1")
    return score


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ModelOutputError("Model output did not contain a JSON object")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            raise ModelOutputError("Model output was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ModelOutputError("Model output must be a JSON object")
    return parsed


def parse_published_date(entry: Any) -> str:
    for key in ("published_parsed", "updated_parsed"):
        if value := entry.get(key):
            return datetime(*value[:6], tzinfo=timezone.utc).date().isoformat()
    for key in ("published", "updated", "pubDate"):
        value = entry.get(key)
        if not value:
            continue
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.date().isoformat()
        except (TypeError, ValueError, OverflowError):
            try:
                return (
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                    .date()
                    .isoformat()
                )
            except (TypeError, ValueError):
                continue
    return ""


def parse_authors(entry: Any) -> list[str]:
    authors: list[str] = []
    for author in entry.get("authors", []) or []:
        name = author.get("name", "") if isinstance(author, dict) else str(author)
        if name and name not in authors:
            authors.append(name.strip())
    if not authors and entry.get("author"):
        authors = [
            part.strip() for part in str(entry.author).split(",") if part.strip()
        ]
    return authors


def paper_from_feed_entry(entry: Any) -> Paper | None:
    title = " ".join(str(entry.get("title", "")).split())
    link = str(entry.get("link", "")).strip()
    if not title or not link:
        return None
    paper_id = canonical_arxiv_id(link)
    canonical_link = f"https://arxiv.org/abs/{paper_id}" if paper_id else link
    abstract = BeautifulSoup(str(entry.get("summary", "")), "html.parser").get_text(
        " ", strip=True
    )
    return Paper(
        title=title,
        link=canonical_link,
        abstract=abstract,
        paper_id=paper_id or canonical_paper_key(Paper(title=title, link=link)),
        authors=parse_authors(entry),
        published_date=parse_published_date(entry),
    )


def is_fresh(paper: Paper, now: datetime, max_age_days: int) -> bool:
    age = paper_age_days(paper, now)
    return age is not None and 0 <= age <= max_age_days


def paper_age_days(paper: Paper, now: datetime) -> int | None:
    """Return whole-day age for a paper with a valid publication date."""
    if not paper.published_date:
        return None
    try:
        published = datetime.fromisoformat(paper.published_date)
    except ValueError:
        return None
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return (now.astimezone(timezone.utc).date() - published.date()).days


def merge_duplicate_papers(first: Paper, second: Paper) -> Paper:
    richer, other = sorted(
        (first, second),
        key=lambda paper: (
            len(paper.abstract),
            len(paper.authors),
            paper.published_date,
            paper.title,
        ),
        reverse=True,
    )
    if not richer.authors:
        richer.authors = list(other.authors)
    if not richer.published_date:
        richer.published_date = other.published_date
    return richer


def candidate_recency_key(paper: Paper) -> tuple[str, str]:
    return (paper.published_date or "0000-00-00", canonical_paper_key(paper))


def select_weekly_candidates(
    ranked: list[Paper],
    now: datetime,
    final_floor: float,
    min_featured: int,
    max_featured: int,
    max_additional: int,
) -> WeeklySelection | None:
    """Select a publishable issue without treating low relevance as an outage."""
    aged: list[tuple[Paper, int]] = []
    for paper in ranked:
        age = paper_age_days(paper, now)
        if age is not None:
            aged.append((paper, age))
    for window_days in SELECTION_WINDOWS_DAYS:
        eligible = [paper for paper, age in aged if 0 <= age <= window_days]
        qualified = [paper for paper in eligible if paper.interest_score >= final_floor]
        if len(qualified) < min_featured:
            continue
        featured = qualified[:max_featured]
        additional = qualified[max_featured : max_featured + max_additional]
        return WeeklySelection(
            featured=featured,
            additional=additional,
            selection_mode=(
                "quality_floor"
                if window_days == SELECTION_WINDOWS_DAYS[0]
                else "expanded_window"
            ),
            source_window_days=window_days,
            valid_candidates=len(aged),
        )

    oldest_window = SELECTION_WINDOWS_DAYS[-1]
    eligible = [paper for paper, age in aged if 0 <= age <= oldest_window]
    if len(eligible) < min_featured:
        return None
    qualified = [paper for paper in eligible if paper.interest_score >= final_floor]
    featured = qualified[:max_featured]
    if len(featured) < min_featured:
        selected_keys = {canonical_paper_key(paper) for paper in featured}
        featured.extend(
            paper
            for paper in eligible
            if canonical_paper_key(paper) not in selected_keys
        )
        featured = featured[: max(min_featured, len(qualified))]
        featured = featured[:max_featured]
    featured_keys = {canonical_paper_key(paper) for paper in featured}
    additional = [
        paper for paper in eligible if canonical_paper_key(paper) not in featured_keys
    ][:max_additional]
    return WeeklySelection(
        featured=featured,
        additional=additional,
        selection_mode="best_available",
        source_window_days=oldest_window,
        valid_candidates=len(aged),
    )


def extract_relevant_sections(html_text: str) -> str:
    """Extract grounded, section-aware source text and stop before references."""
    soup = BeautifulSoup(html_text, "html.parser")
    main = soup.find("main", class_="ltx_page_main") or soup.find(
        "div", class_="ltx_page_main"
    )
    if main is None:
        return ""
    section_patterns = {
        "abstract": ("abstract",),
        "method": (
            "method",
            "methodology",
            "approach",
            "architecture",
            "system design",
        ),
        "evaluation": (
            "result",
            "experiment",
            "evaluation",
            "analysis",
            "benchmark",
        ),
        "limitations": ("limitation", "discussion", "threats to validity"),
        "conclusion": ("conclusion", "future work"),
    }
    selected_parts: list[str] = []
    seen_text: set[str] = set()
    section_sizes = {name: 0 for name in section_patterns}
    active_section: str | None = None

    if abstract_container := main.select_one(".ltx_abstract"):
        abstract_text = " ".join(abstract_container.get_text(" ", strip=True).split())
        if abstract_text:
            selected_parts.append(f"[Abstract]\n{abstract_text}")
            seen_text.add(abstract_text.casefold())
            section_sizes["abstract"] += len(abstract_text)

    for element in main.find_all(["h2", "h3", "p"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        normalized = text.casefold()
        if element.name in {"h2", "h3"}:
            heading = re.sub(r"^\s*[\d.ivx]+(?:\.[\d.ivx]+)*\s*", "", normalized)
            if "reference" in heading or "bibliograph" in heading:
                break
            active_section = next(
                (
                    name
                    for name, patterns in section_patterns.items()
                    if any(pattern in heading for pattern in patterns)
                ),
                None,
            )
            if active_section is not None:
                selected_parts.append(f"[{text}]")
            continue
        if active_section is None or normalized in seen_text:
            continue
        if section_sizes[active_section] >= 12000:
            continue
        selected_parts.append(text)
        seen_text.add(normalized)
        section_sizes[active_section] += len(text)
    return "\n\n".join(selected_parts)[:50000]

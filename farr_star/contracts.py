from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from . import compat  # noqa: F401
from farr.documents import normalize
from farr.parsing import _json_value


REASONING_TYPES = {
    "bridge",
    "comparison",
    "comparison_chain",
    "compositional",
    "inference",
    "temporal",
    "numeric",
    "boolean",
    "other",
}
ANSWER_TYPES = {
    "person",
    "organization",
    "location",
    "work",
    "date",
    "number",
    "boolean",
    "cause",
    "entity",
    "text",
}
MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth",
}


@dataclass
class QuestionContract:
    reasoning_type: str = "other"
    answer_type: str = "entity"
    target: str = "the entity or value explicitly requested by the question"
    expected_hops: int = 2
    requires_comparison: bool = False
    return_compared_entity: bool = False
    allowed_answers: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_list(value: Any, limit: int, max_words: int = 30) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = " ".join(str(item or "").split()).strip(" \"'")
        if not text or len(text.split()) > max_words or text in result:
            continue
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _as_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return fallback


def heuristic_contract(question: str) -> QuestionContract:
    q = " ".join(str(question).split())
    lower = q.lower()
    yes_no = lower.startswith(
        (
            "is ", "are ", "was ", "were ", "do ", "does ", "did ",
            "has ", "have ", "had ", "can ", "could ", "will ", "would ",
        )
    )
    comparison = bool(
        re.search(
            r"\b(earlier|later|older|younger|higher|lower|more|less|"
            r"longer|shorter|before|after|same|both|first|last)\b",
            lower,
        )
    )
    alternatives = []
    if comparison and " or " in lower and not yes_no:
        alternative_match = re.search(
            r",\s*([^,?]{1,120}?)\s+or\s+([^?]{1,120})\??$",
            q,
            flags=re.I,
        )
        if alternative_match:
            possible = [
                alternative_match.group(1).strip(" \"'"),
                alternative_match.group(2).strip(" \"'"),
            ]
            if all(0 < len(item.split()) <= 15 for item in possible):
                alternatives = possible
    which_match = re.match(r"which\s+([a-z][a-z -]{0,30}?)\s+(?:has|had|was|is|did|does)\b", lower)
    target = which_match.group(1).strip() if which_match else ""
    if lower.startswith("why "):
        reasoning_type = "compositional" if "'s " in lower else "inference"
    elif " or " in lower and comparison:
        # This fallback intentionally records only an approximate target.
        # The LLM contract extractor supplies exact explicit alternatives.
        reasoning_type = "comparison_chain" if re.search(
            r"\b(who|whose|director|author|spouse|parent|member|creator|"
            r"founded|born|died)\b",
            lower,
        ) else "comparison"
    elif comparison:
        reasoning_type = "comparison"
    elif yes_no:
        reasoning_type = "boolean"
    elif re.search(r"\b(when|what year|date|before|after)\b", lower):
        reasoning_type = "temporal"
    elif re.search(r"\b(how many|number of|total|difference)\b", lower):
        reasoning_type = "numeric"
    else:
        reasoning_type = "bridge"

    if lower.startswith("why "):
        answer_type = "cause"
        target = "the reason or cause requested by the question"
    elif yes_no:
        answer_type = "boolean"
        alternatives = ["yes", "no"]
        target = "yes or no"
    elif lower.startswith(("who ", "whose ")):
        answer_type = "person"
        target = target or "the requested person"
    elif lower.startswith(("where ",)):
        answer_type = "location"
        target = target or "the requested place"
    elif lower.startswith(("when ", "what year", "on what date")):
        answer_type = "date"
        target = target or "the requested date or year"
    elif lower.startswith(("how many ", "what number", "which number")):
        answer_type = "number"
        target = target or "the requested number"
    elif target and any(
        word in target
        for word in (
            "film", "movie", "book", "novel", "song", "album", "series",
            "season", "episode", "work", "play",
        )
    ):
        answer_type = "work"
    else:
        answer_type = "entity"

    return QuestionContract(
        reasoning_type=reasoning_type,
        answer_type=answer_type,
        target=target or "the entity or value explicitly requested by the question",
        expected_hops=3 if reasoning_type == "comparison_chain" else 2,
        requires_comparison=comparison,
        return_compared_entity=bool(comparison and lower.startswith("which ")),
        allowed_answers=alternatives,
        constraints=[],
        retrieval_queries=[q],
    )


def parse_contract(raw: str, question: str) -> QuestionContract:
    fallback = heuristic_contract(question)
    value = _json_value(raw)
    if not isinstance(value, dict):
        return fallback

    reasoning_type = str(
        value.get("reasoning_type") or fallback.reasoning_type
    ).lower()
    if reasoning_type not in REASONING_TYPES:
        reasoning_type = fallback.reasoning_type
    answer_type = str(value.get("answer_type") or fallback.answer_type).lower()
    if answer_type not in ANSWER_TYPES:
        answer_type = fallback.answer_type
    try:
        expected_hops = int(value.get("expected_hops", fallback.expected_hops))
    except (TypeError, ValueError):
        expected_hops = fallback.expected_hops

    allowed = _clean_list(value.get("allowed_answers"), 4, 15)
    if not allowed:
        allowed = fallback.allowed_answers
    if answer_type == "boolean" and not allowed:
        allowed = ["yes", "no"]
    queries = _clean_list(value.get("retrieval_queries"), 4, 35)
    if not queries:
        queries = fallback.retrieval_queries

    lower_question = " ".join(str(question).lower().split())
    target = " ".join(
        str(value.get("target") or fallback.target).split()
    )[:240]
    requires_comparison = _as_bool(
        value.get("requires_comparison"),
        fallback.requires_comparison,
    )
    return_compared_entity = _as_bool(
        value.get(
            "return_compared_entity",
            fallback.return_compared_entity,
        ),
        fallback.return_compared_entity,
    )

    # Interrogative semantics are hard invariants. An LLM contract may confuse
    # "Why did ... die?" with "When did ... die?"; never allow that mistake to
    # redirect the whole pipeline toward a date.
    if lower_question.startswith("why "):
        answer_type = "cause"
        target = "the reason or cause requested by the question"
        requires_comparison = False
        return_compared_entity = False
        if reasoning_type in {"comparison", "comparison_chain", "temporal"}:
            reasoning_type = (
                "compositional" if "'s " in lower_question else "inference"
            )
    elif lower_question.startswith("where "):
        answer_type = "location"
    elif lower_question.startswith("when "):
        answer_type = "date"
        if (
            reasoning_type in {"comparison", "comparison_chain"}
            and not fallback.requires_comparison
        ):
            reasoning_type = "compositional"
            requires_comparison = False
            return_compared_entity = False
    elif lower_question.startswith(("how many ", "how much ")):
        answer_type = "number"

    return QuestionContract(
        reasoning_type=reasoning_type,
        answer_type=answer_type,
        target=target,
        expected_hops=max(1, min(expected_hops, 5)),
        requires_comparison=requires_comparison,
        return_compared_entity=return_compared_entity,
        allowed_answers=allowed,
        constraints=_clean_list(value.get("constraints"), 6, 30),
        retrieval_queries=queries,
    )


def answer_contract_compliance(
    answer: str,
    contract: QuestionContract,
) -> tuple[bool, str]:
    raw = " ".join(str(answer).split()).strip(" \"'`.")
    candidate = normalize(raw)
    if not candidate:
        return False, "empty answer"

    if contract.allowed_answers:
        normalized_allowed = [normalize(item) for item in contract.allowed_answers]
        if candidate not in normalized_allowed:
            return False, "answer is not one of the explicit return alternatives"

    if contract.answer_type == "boolean":
        return (
            candidate in {"yes", "no"},
            "boolean answer" if candidate in {"yes", "no"} else "expected yes/no",
        )
    if contract.answer_type == "date":
        tokens = set(candidate.split())
        valid = bool(re.search(r"\b\d{3,4}\b", candidate) or tokens & MONTHS)
        return valid, "date-shaped answer" if valid else "expected a date or year"
    if contract.answer_type == "number":
        tokens = set(candidate.split())
        valid = bool(re.search(r"\d", candidate) or tokens & NUMBER_WORDS)
        return valid, "number-shaped answer" if valid else "expected a number"
    if contract.answer_type == "cause":
        date_signal = bool(
            re.search(r"\b\d{3,4}\b", candidate)
            or set(candidate.split()) & MONTHS
        )
        causal_signal = bool(
            re.search(
                r"\b(because|due to|caused by|from|of|as a result|"
                r"illness|disease|stroke|cancer|injur\w*|accident|attack|"
                r"failure|infection|suicide|murder|killed)\b",
                candidate,
            )
        )
        if date_signal and not causal_signal:
            return False, "expected a cause, not a death date"
        return True, "cause-shaped answer"
    if len(candidate.split()) > 18:
        return False, "answer is not concise"
    return True, "no deterministic contract violation"

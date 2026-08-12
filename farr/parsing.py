from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _json_value(raw: str) -> Any:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def parse_plan(raw: str, question: str, max_hops: int) -> List[str]:
    value = _json_value(raw)
    items: Any = []
    if isinstance(value, dict):
        items = value.get("subquestions") or value.get("hops") or value.get("plan") or []
    elif isinstance(value, list):
        items = value

    subquestions = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("subquestion") or item.get("question") or item.get("goal")
        else:
            text = item
        if text and str(text).strip():
            subquestions.append(str(text).strip())

    if not subquestions:
        for line in re.split(r"[\n;]+", str(raw)):
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if line and len(line.split()) > 2 and not line.startswith(("{", "}")):
                subquestions.append(line)

    if not subquestions:
        subquestions = [question]

    result = []
    for subquestion in subquestions:
        if subquestion not in result:
            result.append(subquestion)
    return result[:max_hops]


def parse_queries(raw: str, fallbacks: List[str], max_queries: int) -> List[str]:
    value = _json_value(raw)
    items: Any = []
    if isinstance(value, dict):
        items = value.get("queries") or value.get("search_queries") or []
    elif isinstance(value, list):
        items = value

    if not items:
        items = re.split(r"[\n;]+", str(raw))

    queries = []
    for item in [*items, *fallbacks]:
        if isinstance(item, dict):
            item = item.get("query") or item.get("text")
        query = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", str(item or "")).strip()
        query = query.strip("\"'")
        if not query or query.lower() in {"none", "n/a", "no query"}:
            continue
        if query not in queries:
            queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def parse_hop_answer(raw: str) -> Dict[str, str]:
    value = _json_value(raw)
    if isinstance(value, dict):
        answer = value.get("answer") or value.get("intermediate_answer") or ""
        missing = value.get("missing_information") or value.get("missing") or ""
        return {"answer": str(answer).strip(), "missing_information": str(missing).strip()}

    text = str(raw).strip()
    missing_match = re.search(r"missing(?: information)?\s*:\s*(.+)", text, re.I)
    answer_match = re.search(r"(?:intermediate_)?answer\s*:\s*(.+)", text, re.I)
    return {
        "answer": (answer_match.group(1) if answer_match else text).strip(),
        "missing_information": (missing_match.group(1) if missing_match else "").strip(),
    }


def parse_verification(raw: str) -> Dict[str, str]:
    value = _json_value(raw)
    if isinstance(value, dict):
        label = str(value.get("label") or value.get("verdict") or "UNCERTAIN").upper()
        rationale = value.get("rationale") or value.get("reason") or ""
        correction = value.get("correction") or value.get("suggested_answer") or ""
    else:
        text = str(raw)
        upper = text.upper()
        if "UNSUPPORTED" in upper or "REFUTED" in upper or "CONTRADICT" in upper:
            label = "UNSUPPORTED"
        elif "SUPPORTED" in upper and "UNSUPPORTED" not in upper:
            label = "SUPPORTED"
        else:
            label = "UNCERTAIN"
        rationale, correction = text, ""

    if label not in {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}:
        label = "UNCERTAIN"
    return {
        "label": label,
        "rationale": str(rationale).strip(),
        "correction": str(correction).strip(),
    }


def parse_revised_answer(raw: str) -> str:
    value = _json_value(raw)
    if isinstance(value, dict):
        answer = value.get("answer") or value.get("revised_answer")
        if answer is not None:
            return str(answer).strip()
    return str(raw).strip()

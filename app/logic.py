# app/logic.py
import json
import re
from collections import Counter
from typing import Dict, List, Tuple

from pydantic import ValidationError

from .schema import Transaction, PageExtraction, FinalReport

# -------- Robust sanitization / repair helpers --------

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
THINK_TAG_RE = re.compile(r"</?think[^>]*>", re.IGNORECASE)

def _find_balanced_json(text: str) -> str | None:
    """
    Return the first balanced top-level JSON array or object found in text.
    """
    for opener, closer in ("[]", "{}"):
        start = text.find(opener[0])
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if ch == '"' and not esc:
                    in_str = not in_str
                if ch == "\\" and not esc:
                    esc = True
                    continue
                esc = False
                if in_str:
                    continue
                if ch == opener[0]:
                    depth += 1
                elif ch == closer[0]:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
            start = text.find(opener[0], start + 1)
    return None

def _wrap_multiple_top_objects(text: str) -> str | None:
    """
    Wrap multiple top-level JSON objects into an array if they are printed one after another.
    """
    objs = []
    i = 0
    s = text.strip()
    while True:
        j = s.find("{", i)
        if j == -1:
            break
        depth = 0
        in_str = False
        esc = False
        found = False
        for k in range(j, len(s)):
            ch = s[k]
            if ch == '"' and not esc:
                in_str = not in_str
            if ch == "\\" and not esc:
                esc = True
                continue
            esc = False
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objs.append(s[j : k + 1])
                    i = k + 1
                    found = True
                    break
        if not found:
            break
    if len(objs) >= 2:
        return "[" + ",".join(objs) + "]"
    return None

def sanitize_model_json(text: str) -> str:
    # Coerce None to empty string
    if text is None:
        text = ""
    # Strip code fences and <think> traces
    text = FENCE_RE.sub("", str(text).strip())
    text = THINK_TAG_RE.sub("", text)

    # Already valid?
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Try to find one balanced JSON block
    cand = _find_balanced_json(text)
    if cand:
        return cand

    # Try wrapping multiple top-level objects
    wrapped = _wrap_multiple_top_objects(text)
    if wrapped:
        return wrapped

    # Last resort: convert bare single quotes to double quotes
    soft = re.sub(r"(?<!\\)'", '"', text)
    cand = _find_balanced_json(soft)
    if cand:
        return cand

    # Give up; let json.loads raise for diagnostics
    return text

# -------- Core pipeline logic --------

def chunk_pages(pages_text: List[str], chunk_size: int = 2) -> List[Dict[int, str]]:
    chunks: List[Dict[int, str]] = []
    for i in range(0, len(pages_text), chunk_size):
        group = {i + j: pages_text[i + j] for j in range(min(chunk_size, len(pages_text) - i))}
        chunks.append(group)
    return chunks

def parse_llm_json(raw: str) -> List[PageExtraction]:
    raw = sanitize_model_json(raw)
    data = json.loads(raw)

    # Accept dict (single page) or list (multiple pages) or {"pages":[...]}
    if isinstance(data, dict) and "pages" in data and isinstance(data["pages"], list):
        data = data["pages"]
    elif isinstance(data, dict):
        data = [data]

    pages: List[PageExtraction] = []
    for obj in data:
        # If only "transactions" present, synthesize a page_index
        if "page_index" not in obj and "transactions" in obj and isinstance(obj["transactions"], list):
            obj["page_index"] = -1
        pages.append(PageExtraction(**obj))
    return pages

def aggregate(pages: List[PageExtraction]) -> FinalReport:
    txs: List[Transaction] = []
    for p in pages:
        for t in p.transactions:
            txs.append(Transaction(**(t if isinstance(t, dict) else t.dict())))
    total_credits = sum(t.amount for t in txs if t.type == "credit")
    currency_guess = next((t.currency for t in txs if t.currency), None)
    return FinalReport(
        total_credits=round(total_credits, 2),
        total_income=round(total_credits, 2),  # alias
        currency_guess=currency_guess,
        transactions=txs,
    )

def safe_parse_and_aggregate(raw: str) -> Tuple[FinalReport | None, str | None]:
    try:
        pages = parse_llm_json(raw)
        report = aggregate(pages)
        return report, None
    except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"

def compare_models(reports: Dict[str, FinalReport]) -> Dict:
    totals = {m: r.total_credits for m, r in reports.items()}
    values = list(totals.values())
    if not values:
        return {"agreement": False, "reason": "no valid reports"}
    lo, hi = min(values), max(values)
    agreement = (hi - lo) <= 1.0  # within 1 unit -> "agree"
    freq = Counter(round(v, 2) for v in values)
    winner, _ = freq.most_common(1)[0]
    return {
        "agreement": agreement,
        "range": {"min": lo, "max": hi, "spread": round(hi - lo, 2)},
        "majority_total_credits": winner,
        "votes": dict(freq),
        "per_model_totals": totals,
    }

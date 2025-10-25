import httpx
from typing import Dict, List

from .config import OLLAMA_URL, JSON_FORMAT_OPTION

SYSTEM = "You are a strict JSON extractor for bank statements; return only VALID JSON matching the schema, focus on INCOMING FUNDS (type \"credit\"), first search the page/document text for a printed total labelled like 'Total Credit', 'TOTAL CREDIT AMT', or 'Total Credits' (case-insensitive) and use that value as the total credit without computing; if none is present, then compute the total credit by summing 'credit' transactions; never include prose, chain-of-thought, or code fences."

SCHEMA_HINT = r"""
Return JSON exactly like:
[
  {
    "page_index": <int>,
    "transactions": [
      {
        "date": "YYYY-MM-DD or as-seen",          // posting/transaction date (as shown)
        "value_date": "YYYY-MM-DD or as-seen",    // optional; include if present
        "description": "text",                    // full narrative for the row
        "channel": "text",                        // optional; e.g., 'NIP Transfer', 'Online Banking', 'POS', 'Others'
        "doc_no": "text",                         // optional; include if a doc/ref number column exists
        "amount": 12345.67,                       // POSITIVE number, strip commas/currency signs
        "currency": "NGN",                        // optional; NGN/₦ if visible; else omit
        "type": "credit"                          // 'credit' for incoming funds, 'debit' for outgoing
      }
    ],
    "page_credit_sum": 123456.78,                 // REQUIRED: sum of 'credit' amounts on THIS page (compute if not printed)
    "page_debit_sum": 2345.67                     // OPTIONAL: sum of 'debit' amounts on THIS page (compute if easy)
  }
]

Extraction rules (strict):
- Focus on INCOMING FUNDS: ensure ALL 'credit' rows are captured accurately.
- Each row must have EXACTLY ONE 'amount' and ONE 'type':
  • If a 'credit' column exists (e.g., 'Pay In' or 'CR'), use that non-empty value → type: "credit".
- Numeric cleanup: remove commas and currency symbols before parsing; amounts must be positive (do NOT include minus signs).
- Map common aliases/keywords to type:
  • Incoming → "credit": CR, CREDIT, PAY IN, IN, DEPOSIT, NIP/NXG/UIP/USSD "Trf from"/"Transfer from", REFUND, REVERSAL CR, INTEREST, CASH DEPOSIT.
- Columns you may see:
  • Layout A (e.g., “Pay In / Pay Out / Balance”): take 'Pay In' → credit, 'Pay Out' → debit.
  • If both a posting/transaction date and a value date exist, set 'date' = posting/transaction date and 'value_date' accordingly.
  • Optional fields 'channel' and 'doc_no' should be included when those columns exist; omit otherwise.
- Ignore non-transaction lines: headers/footers, URLs, “Opening Balance”, “Closing Balance”, column headings, page numbers, and decorative totals—EXCEPT the statement’s printed grand totals.
- When the document prints a grand total like “TOTAL CREDIT AMT = 1,058,056.81”:
  • Add it to the LAST page object under a 'document_totals' key:
    "document_totals": { "total_credit_reported": 1058056.81}
  • Only include keys that actually appear (don't invent totals).
- Always compute 'page_credit_sum' from the extracted transactions on that page (even if a page subtotal is printed).
- Do NOT include running balances as transactions.
- JSON ONLY. No prose, no code fences, no extra keys outside the schema.

Document-level Total Credit selection rule (strict):
  - FIRST: search the entire page/document text for printed labels:
    'Total Credit', 'TOTAL CREDIT AMT', 'Total Credits' (case-insensitive).
    If found, set document_total_credit to that printed number (do NOT compute),
    set document_total_credit_method = "printed",
    and set document_total_credit_source_label to the exact matched label text.
  - ELSE: compute by summing all 'credit' transaction amounts across pages,
    set document_total_credit_method = "computed".
"""


PROMPT_TEMPLATE = """
You will receive 1–3 pages of a bank statement. Two common layouts may appear:
- Layout A: columns include 'Pay In', 'Pay Out', 'Balance'.
Extract ONLY transactions, focusing on INCOMING FUNDS (credits). Return JSON per the schema below.

{schema_hint}

PAGES:
{pages_block}
"""


def build_prompt(pages: Dict[int, str]) -> str:
    pages_block = "\n\n".join([f"--- PAGE {i} ---\n{txt}" for i, txt in pages.items()])
    return PROMPT_TEMPLATE.format(schema_hint=SCHEMA_HINT, pages_block=pages_block)

async def chat_ollama_model(model: str, user_content: str) -> str:
    """
    Calls Ollama /api/chat and returns plain text content.
    Works with both {"message":{"content":...}} and {"response":...} payloads.
    Always returns a string; raises only for network/HTTP errors.
    """
    options = {"temperature": 0}
    if JSON_FORMAT_OPTION:
        options["format"] = "json"
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "options": options,
            },
        )
        r.raise_for_status()
        data = r.json() or {}
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                content = data["message"].get("content")
                if isinstance(content, str):
                    return content
            if "response" in data and isinstance(data["response"], str):
                return data["response"]
        return json.dumps(data)

REPAIR_INSTRUCTION = (
    "Your previous response was not valid JSON. "
    "Respond again with VALID JSON ONLY that matches the schema. "
    "If no transactions are present, return an empty array []."
)

async def chat_ollama_model_retry_json(model: str, original_user_content: str, previous_output: str) -> str:
    options = {"temperature": 0}
    if JSON_FORMAT_OPTION:
        options["format"] = "json"
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": original_user_content},
                    {"role": "assistant", "content": previous_output},
                    {"role": "user", "content": REPAIR_INSTRUCTION},
                ],
                "stream": False,
                "options": options,
            },
        )
        r.raise_for_status()
        data = r.json() or {}
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                content = data["message"].get("content")
                if isinstance(content, str):
                    return content
            if "response" in data and isinstance(data["response"], str):
                return data["response"]
        return json.dumps(data)

async def extract_page_chunk_multi(models: List[str], pages: Dict[int, str]) -> Dict[str, str]:
    """
    Runs all models concurrently on the same chunk.
    Always returns a dict {model: raw_text_or_error_json}.
    """
    import asyncio
    prompt = build_prompt(pages)
    out: Dict[str, str] = {}
    try:
        tasks = [chat_ollama_model(m, prompt) for m in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for m, res in zip(models, results):
            if isinstance(res, Exception):
                out[m] = f'{{"error":"{type(res).__name__}: {str(res)}"}}'
            else:
                out[m] = res if isinstance(res, str) else json.dumps(res)
    except Exception as e:
        for m in models:
            out[m] = f'{{"error":"{type(e).__name__}: {str(e)}"}}'
    return out
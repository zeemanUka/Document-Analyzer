# app/schema.py
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    currency: Optional[str] = None
    type: Literal["credit", "debit"]

class PageExtraction(BaseModel):
    page_index: int
    transactions: List[Transaction] = Field(default_factory=list)

class FinalReport(BaseModel):
    total_credits: float
    total_income: float
    currency_guess: Optional[str] = None
    transactions: List[Transaction]

class MultiModelReport(BaseModel):
    by_model: Dict[str, FinalReport] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    comparison: Dict = Field(default_factory=dict)
    meta: Dict = Field(default_factory=dict)   # job_id, chunks, timings, history

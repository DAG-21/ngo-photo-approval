"""
FastAPI Service for the NGO Photo Approval Engine.
Exposes:
- `GET /health`: Health & configuration check
- `POST /api/v1/evaluate`: Evaluates an uploaded photograph using Prompt 4 + Dynamic Few-Shot Retrieval
- `POST /api/v1/human-feedback`: Records NGO reviewer override/confirmation without overwriting model annotations
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.schemas import ProductionEvaluationOutput

app = FastAPI(
    title="NGO Photo Approval Engine API",
    description="Evidence-based NGO social media photo evaluation engine with 16 mandatory/advisory rules, dynamic few-shot retrieval, and human-in-the-loop review routing.",
    version="2.0.0",
)


class HumanReviewSubmission(BaseModel):
    image_id: str
    reviewer_id: str
    final_decision: str  # APPROVE | REJECT
    confirmed_tags: List[str] = []
    confirmed_rejection_reasons: List[str] = []
    reviewer_notes: Optional[str] = None


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "ngo-photo-approval-engine", "version": "2.0.0"}


@app.post("/api/v1/human-feedback")
def record_human_feedback(payload: HumanReviewSubmission) -> Dict[str, Any]:
    """
    Stores human reviewer confirmation or override in a dedicated audit log so that
    model observations and confirmed NGO decisions remain strictly separated.
    """
    base_dir = Path(__file__).resolve().parents[2]
    feedback_file = base_dir / "outputs/final_decisions/human_reviewer_feedback.jsonl"
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload.to_dict()) + "\n")
    return {"status": "recorded", "image_id": payload.image_id, "final_decision": payload.final_decision}

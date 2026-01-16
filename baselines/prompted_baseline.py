import json
import os
from typing import Dict

from openai import OpenAI


_SYSTEM_PROMPT = """You are a risk-gating assistant. Given a customer message, output ONLY a JSON object with:
- override_required: boolean
- reason_codes: array of strings
- confidence: number between 0 and 1
- recommended_next_step: string

Allowed reason_codes: LEGAL_RISK, PRIVACY_RISK, SAFETY_RISK, FRAUD_RISK, SECURITY_RISK, OTHER_RISK, NONE.
If override_required is false, reason_codes must be ["NONE"] and recommended_next_step must be "automated_resolution".
Return only valid JSON. No extra text."""


def _client() -> OpenAI:
    return OpenAI()


def predict(text: str, model: str | None = None) -> Dict[str, object]:
    model_name = model or os.getenv("OPENAI_BASE_MODEL", "gpt-4o-mini")
    client = _client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


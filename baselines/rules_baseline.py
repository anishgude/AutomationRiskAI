import re
from typing import Dict, List


_RISK_RULES = {
    "LEGAL_RISK": [
        r"\blawyer\b",
        r"\blegal\b",
        r"\bsue\b",
        r"\blawsuit\b",
        r"\battorney\b",
        r"\bcourt\b",
    ],
    "PRIVACY_RISK": [
        r"\bdelete\b.*\bdata\b",
        r"\bpersonal information\b",
        r"\bprivacy\b",
        r"\bgdpr\b",
        r"\bccpa\b",
        r"\bdata removal\b",
    ],
    "SAFETY_RISK": [
        r"\bsuicide\b",
        r"\bself-harm\b",
        r"\bkill myself\b",
        r"\bthreat\b",
        r"\bviolence\b",
        r"\bweapon\b",
    ],
    "FRAUD_RISK": [
        r"\bfraud\b",
        r"\bscam\b",
        r"\bchargeback\b",
        r"\bstolen\b",
        r"\bidentity theft\b",
    ],
    "SECURITY_RISK": [
        r"\bhack\b",
        r"\bbreach\b",
        r"\bcompromised\b",
        r"\baccount takeover\b",
    ],
    "OTHER_RISK": [
        r"\bpress\b",
        r"\bmedia\b",
        r"\bregulator\b",
        r"\bcomplaint\b",
    ],
}

_PRIORITY = [
    "LEGAL_RISK",
    "SAFETY_RISK",
    "PRIVACY_RISK",
    "FRAUD_RISK",
    "SECURITY_RISK",
    "OTHER_RISK",
]

_NEXT_STEP = {
    "LEGAL_RISK": "escalate_to_legal_team",
    "SAFETY_RISK": "escalate_to_safety_team",
    "PRIVACY_RISK": "escalate_to_privacy_team",
    "FRAUD_RISK": "escalate_to_fraud_team",
    "SECURITY_RISK": "escalate_to_security_team",
    "OTHER_RISK": "manual_review",
    "NONE": "automated_resolution",
}


def _match_reason_codes(text: str) -> List[str]:
    text_l = text.lower()
    reason_codes = []
    for code, patterns in _RISK_RULES.items():
        for pattern in patterns:
            if re.search(pattern, text_l):
                reason_codes.append(code)
                break
    return reason_codes


def predict(text: str) -> Dict[str, object]:
    reason_codes = _match_reason_codes(text)
    override_required = len(reason_codes) > 0

    if not override_required:
        return {
            "override_required": False,
            "reason_codes": ["NONE"],
            "confidence": 0.55,
            "recommended_next_step": _NEXT_STEP["NONE"],
        }

    reason_codes_sorted = sorted(
        reason_codes, key=lambda code: _PRIORITY.index(code)
    )
    primary_reason = reason_codes_sorted[0]
    confidence = min(0.9 + 0.02 * (len(reason_codes_sorted) - 1), 0.97)

    return {
        "override_required": True,
        "reason_codes": reason_codes_sorted,
        "confidence": round(confidence, 2),
        "recommended_next_step": _NEXT_STEP[primary_reason],
    }


def predict_from_record(record: Dict[str, object]) -> Dict[str, object]:
    messages = record.get("messages", [])
    user_text = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_text = msg.get("content", "")
            break
    return predict(user_text)


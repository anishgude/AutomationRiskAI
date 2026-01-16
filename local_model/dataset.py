import json
from pathlib import Path
from typing import List, Tuple


def _extract_user_text(record: dict) -> str:
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _extract_label(record: dict) -> bool:
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "{}")
            parsed = json.loads(content)
            return bool(parsed.get("override_required", False))
    raise ValueError("No assistant label found in record")


def load_dataset(path: str | Path) -> Tuple[List[str], List[int]]:
    path = Path(path)
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts.append(_extract_user_text(record))
            labels.append(1 if _extract_label(record) else 0)
    return texts, labels

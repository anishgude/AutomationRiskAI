import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from baselines import prompted_baseline, rules_baseline
from eval.metrics import precision_recall_fnr


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_user_text(record: Dict[str, object]) -> str:
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_label(record: Dict[str, object]) -> bool:
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "{}")
            parsed = json.loads(content)
            return bool(parsed.get("override_required", False))
    raise ValueError("No assistant label found in record")


def evaluate_predictions(
    labels: List[bool], predictions: List[bool]
) -> Dict[str, object]:
    metrics = precision_recall_fnr(labels, predictions)
    metrics["total"] = len(labels)
    return metrics


def run_rules(records: Iterable[Dict[str, object]]) -> List[bool]:
    preds = []
    for record in records:
        text = extract_user_text(record)
        result = rules_baseline.predict(text)
        preds.append(bool(result.get("override_required", False)))
    return preds


def run_prompted(records: Iterable[Dict[str, object]], model: str) -> List[bool]:
    preds = []
    for record in records:
        text = extract_user_text(record)
        result = prompted_baseline.predict(text, model=model)
        preds.append(bool(result.get("override_required", False)))
    return preds


def write_results(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "precision",
        "recall",
        "false_negative_rate",
        "tp",
        "fp",
        "tn",
        "fn",
        "total",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/eval.jsonl",
        help="Path to eval JSONL",
    )
    parser.add_argument(
        "--out",
        default="results/results.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.data))
    if args.limit:
        records = records[: args.limit]

    labels = [extract_label(r) for r in records]

    results = []

    rules_preds = run_rules(records)
    rules_metrics = evaluate_predictions(labels, rules_preds)
    results.append({"model": "rules_baseline", **rules_metrics})

    base_model = os.getenv("OPENAI_BASE_MODEL", "gpt-4o-mini")
    prompted_preds = run_prompted(records, model=base_model)
    prompted_metrics = evaluate_predictions(labels, prompted_preds)
    results.append({"model": f"prompted_baseline:{base_model}", **prompted_metrics})

    ft_model = os.getenv("OPENAI_FT_MODEL")
    if not ft_model:
        raise RuntimeError("OPENAI_FT_MODEL is not set")
    ft_preds = run_prompted(records, model=ft_model)
    ft_metrics = evaluate_predictions(labels, ft_preds)
    results.append({"model": f"fine_tuned:{ft_model}", **ft_metrics})

    write_results(Path(args.out), results)


if __name__ == "__main__":
    main()


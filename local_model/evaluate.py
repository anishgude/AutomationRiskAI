import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from local_model.dataset import load_dataset
from local_model.model import MLPClassifier


def _load_artifacts(model_dir: str) -> Tuple[MLPClassifier, object, float]:
    model_dir_path = Path(model_dir)
    with (model_dir_path / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    with (model_dir_path / "vectorizer.pkl").open("rb") as f:
        vectorizer = pickle.load(f)

    model = MLPClassifier(
        input_dim=config["input_dim"], hidden_dims=config["hidden_dims"]
    )
    state = torch.load(model_dir_path / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, vectorizer, float(config.get("threshold", 0.5))


def _metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for y, p in zip(labels, preds):
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 1:
            fp += 1
        elif y == 1 and p == 0:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_negative_rate": round(fnr, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": tp + tn + fp + fn,
    }


def evaluate_model(
    eval_path: str, model_dir: str, out_csv: str
) -> Dict[str, float]:
    texts, labels = load_dataset(eval_path)
    model, vectorizer, threshold = _load_artifacts(model_dir)

    features = vectorizer.transform(texts).toarray().astype(np.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(features))).numpy()
    preds = (probs >= threshold).astype(int).tolist()

    metrics = _metrics(labels, preds)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "label", "probability", "prediction"]
        )
        writer.writeheader()
        for text, label, prob, pred in zip(texts, labels, probs, preds):
            writer.writerow(
                {
                    "text": text,
                    "label": label,
                    "probability": round(float(prob), 6),
                    "prediction": pred,
                }
            )

    print(
        "accuracy={accuracy} precision={precision} recall={recall} "
        "false_negative_rate={false_negative_rate}".format(**metrics)
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/eval.jsonl")
    parser.add_argument("--model-dir", default="local_model/artifacts")
    parser.add_argument("--out", default="local_model/results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(args.eval, args.model_dir, args.out)


if __name__ == "__main__":
    main()

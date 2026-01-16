from typing import Dict, List


def confusion_matrix(y_true: List[bool], y_pred: List[bool]) -> Dict[str, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif truth and not pred:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_fnr(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tp = cm["tp"]
    fp = cm["fp"]
    fn = cm["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_negative_rate": round(fnr, 4),
        **cm,
    }


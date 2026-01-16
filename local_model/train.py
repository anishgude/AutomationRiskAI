import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_model.dataset import load_dataset
from local_model.model import MLPClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def vectorize_texts(
    texts: list[str], max_features: int
) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        lowercase=True,
    )
    features = vectorizer.fit_transform(texts)
    return vectorizer, features.toarray().astype(np.float32)


def train_model(
    train_path: str,
    model_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_features: int,
    hidden_dims: list[int],
    seed: int,
) -> dict:
    set_seed(seed)
    texts, labels = load_dataset(train_path)
    vectorizer, features = vectorize_texts(texts, max_features=max_features)
    labels_np = np.array(labels, dtype=np.float32)

    x_tensor = torch.from_numpy(features)
    y_tensor = torch.from_numpy(labels_np)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPClassifier(input_dim=features.shape[1], hidden_dims=hidden_dims)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir_path / "model.pt")
    with (model_dir_path / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    config = {
        "input_dim": features.shape[1],
        "hidden_dims": hidden_dims,
        "max_features": max_features,
        "threshold": 0.5,
    }
    with (model_dir_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--model-dir", default="local_model/artifacts")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden_dims = [128, 64]
    train_model(
        train_path=args.train,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_features=args.max_features,
        hidden_dims=hidden_dims,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

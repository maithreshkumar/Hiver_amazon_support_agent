from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from hiver_support.config import load_yaml
from hiver_support.intents.taxonomy import load_approved_taxonomy


def train(training_config_path: str | Path = "configs/training.yaml") -> dict[str, float]:
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError('Install neural dependencies with: python -m pip install -e ".[phase2,dev]"') from exc
    config = load_yaml(training_config_path)
    taxonomy = load_approved_taxonomy(config["taxonomy_path"])
    labels = [item.name for item in taxonomy]
    label_to_id = {label: index for index, label in enumerate(labels)}
    data = pd.read_parquet(config["weak_labels_path"])
    data = data.loc[data["predicted_intent"].isin(labels)].copy()
    train_frame, validation_frame = train_test_split(
        data,
        test_size=float(config["classifier"]["validation_fraction"]),
        random_state=int(config["classifier"]["random_seed"]),
        stratify=data["predicted_intent"],
    )
    model_name = str(config["classifier"]["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class TextDataset(Dataset):
        def __init__(self, frame: pd.DataFrame) -> None:
            self.encodings = tokenizer(
                frame["customer_text"].astype(str).tolist(),
                truncation=True,
                padding=True,
                max_length=int(config["classifier"]["max_length"]),
            )
            self.labels = [label_to_id[value] for value in frame["predicted_intent"]]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )

    def metrics(result: object) -> dict[str, float]:
        predicted = np.asarray(result.predictions).argmax(axis=1)
        expected = np.asarray(result.label_ids)
        return {
            "accuracy": float(accuracy_score(expected, predicted)),
            "macro_f1": float(f1_score(expected, predicted, average="macro")),
            "weighted_f1": float(f1_score(expected, predicted, average="weighted")),
        }

    model_dir = Path(config["intent_model_dir"])
    arguments = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=float(config["classifier"]["epochs"]),
        per_device_train_batch_size=int(config["classifier"]["train_batch_size"]),
        per_device_eval_batch_size=int(config["classifier"]["evaluation_batch_size"]),
        learning_rate=float(config["classifier"]["learning_rate"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=[],
        seed=int(config["classifier"]["random_seed"]),
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=TextDataset(train_frame),
        eval_dataset=TextDataset(validation_frame),
        compute_metrics=metrics,
    )
    trainer.train()
    evaluated = trainer.evaluate()
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    summary = {
        key.removeprefix("eval_"): float(value)
        for key, value in evaluated.items()
        if key.startswith("eval_") and isinstance(value, (float, int))
    }
    metadata = {
        "version": datetime.now(UTC).strftime("weak-ai-%Y%m%dT%H%M%SZ"),
        "base_model": model_name,
        "labels": labels,
        "training_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "metrics_against_weak_labels": summary,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary

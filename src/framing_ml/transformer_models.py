# src/framing_ml/transformer_models.py
# Fine-tuning of distilbert-base-multilingual-cased for framing classification.
#
# Uses text_linguistic (not text_ml) because the transformer tokenizer
# handles punctuation and casing itself — sending pre-cleaned text would
# strip information the model relies on.
# The caller (run_framing_ml_models.py) passes TEXT_COL = "text_linguistic".

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

MODEL_NAME   = "distilbert-base-multilingual-cased"
RANDOM_STATE = 42
MAX_LENGTH   = 256


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Candidate configs
# ---------------------------------------------------------------------------

def get_transformer_candidate_configs() -> list[dict]:
    return [
        {"learning_rate": 2e-5, "num_train_epochs": 3,
         "weight_decay": 0.01,  "train_batch_size": 4, "eval_batch_size": 8},
        {"learning_rate": 3e-5, "num_train_epochs": 3,
         "weight_decay": 0.01,  "train_batch_size": 4, "eval_batch_size": 8},
        {"learning_rate": 2e-5, "num_train_epochs": 5,
         "weight_decay": 0.01,  "train_batch_size": 4, "eval_batch_size": 8},
    ]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_binary(y_true, y_pred) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
    }


def evaluate_multiclass(y_true, y_pred) -> dict:
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "macro_f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def metric_to_optimize(task_type: str) -> str:
    return "f1" if task_type == "binary" else "macro_f1"


def evaluate_task(y_true, y_pred, task_type: str) -> dict:
    return evaluate_binary(y_true, y_pred) if task_type == "binary" else evaluate_multiclass(y_true, y_pred)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def build_label_mapping(series: pd.Series) -> tuple[dict, dict]:
    labels   = sorted(series.dropna().unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# Dataset encoding
# ---------------------------------------------------------------------------

def encode_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    tokenizer,
    label2id: dict,
) -> Dataset:
    tmp = df[[text_col, label_col]].copy()
    tmp[text_col] = tmp[text_col].fillna("").astype(str)
    tmp["labels"] = tmp[label_col].map(label2id)

    dataset = Dataset.from_pandas(tmp[[text_col, "labels"]], preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize_batch, batched=True)


def build_compute_metrics(task_type: str):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return evaluate_task(labels, preds, task_type)
    return compute_metrics


# ---------------------------------------------------------------------------
# TrainingArguments builder
# ---------------------------------------------------------------------------

def build_training_args(output_dir: Path, cfg: dict, do_eval: bool) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=RANDOM_STATE,
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
        disable_tqdm=False,
        eval_strategy="epoch" if do_eval else "no",
    )
    return TrainingArguments(**kwargs)


# ---------------------------------------------------------------------------
# Tune
# ---------------------------------------------------------------------------

def tune_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    task_type: str,
    run_dir: Path,
) -> dict:
    device = get_torch_device()
    logger.info(f"Transformer device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label2id, id2label = build_label_mapping(
        pd.concat([train_df[label_col], val_df[label_col]], ignore_index=True)
    )
    num_labels = len(label2id)

    train_ds      = encode_dataframe(train_df, text_col, label_col, tokenizer, label2id)
    val_ds        = encode_dataframe(val_df,   text_col, label_col, tokenizer, label2id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = build_compute_metrics(task_type)
    target_metric   = metric_to_optimize(task_type)

    best_score    = -1.0
    best_cfg      = None
    best_metrics  = None
    best_dir: Path | None = None
    search_results: list[dict] = []

    for idx, cfg in enumerate(get_transformer_candidate_configs(), start=1):
        trial_dir = Path(run_dir) / f"trial_{idx}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_labels,
            label2id=label2id, id2label=id2label,
        ).to(device)

        trainer = Trainer(
            model=model,
            args=build_training_args(trial_dir, cfg, do_eval=True),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        raw_metrics = trainer.evaluate()

        val_metrics = {
            k.replace("eval_", ""): float(v)
            for k, v in raw_metrics.items()
            if k.startswith("eval_")
        }

        search_results.append({
            "model_name":  "bert",
            "label_col":   label_col,
            "task_type":   task_type,
            "params":      cfg,
            "val_metrics": val_metrics,
        })
        logger.info(f"  Trial {idx} val_{target_metric}={val_metrics.get(target_metric, '?'):.3f}")

        if val_metrics.get(target_metric, -1) > best_score:
            best_score   = val_metrics[target_metric]
            best_cfg     = cfg
            best_metrics = val_metrics
            best_dir     = trial_dir

    if best_cfg is None:
        raise RuntimeError("No valid configuration found during transformer tuning.")

    return {
        "best_config":      best_cfg,
        "best_val_metrics": best_metrics,
        "search_results":   search_results,
        "best_trial_dir":   str(best_dir),
        "label2id":         label2id,
        "id2label":         id2label,
    }


# ---------------------------------------------------------------------------
# Final fit
# ---------------------------------------------------------------------------

def fit_final_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    task_type: str,
    best_config: dict,
    label2id: dict,
    id2label: dict,
    run_dir: Path,
) -> tuple[Path, object]:
    device    = get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_labels = len(label2id)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_ds = encode_dataframe(train_val_df, text_col, label_col, tokenizer, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels,
        label2id=label2id, id2label=id2label,
    ).to(device)

    final_dir = Path(run_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=build_training_args(final_dir, best_config, do_eval=False),
        train_dataset=train_val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    return final_dir, tokenizer


# ---------------------------------------------------------------------------
# Evaluate on test
# ---------------------------------------------------------------------------

def evaluate_final_transformer(
    model_dir: Path,
    tokenizer,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    task_type: str,
    label2id: dict,
) -> tuple[dict, np.ndarray]:
    device   = get_torch_device()
    id2label = {idx: label for label, idx in label2id.items()}

    model    = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    test_ds  = encode_dataframe(test_df, text_col, label_col, tokenizer, label2id)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(model_dir) / "eval_tmp"),
            per_device_eval_batch_size=8,
            report_to=[],
            dataloader_num_workers=0,
            fp16=False, bf16=False,
        ),
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    pred_output  = trainer.predict(test_ds)
    y_pred_ids   = np.argmax(pred_output.predictions, axis=1)
    y_pred       = [id2label[idx] for idx in y_pred_ids]
    y_true       = test_df[label_col].tolist()

    return evaluate_task(y_true, y_pred, task_type), np.array(y_pred)
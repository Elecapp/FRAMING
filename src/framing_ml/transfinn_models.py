# src/framing_ml/transfinn_models.py
# TransFINN: Transparent Feature-Integrated Neural Network (PyTorch implementation).
#
# Reference: Awan et al., "TransFINN: Transparent Feature-Integrated Neural
# Network for Text Feature Selection and Classification"
# https://github.com/saifurrehmanawan/TransFINN
#
# Architecture
# ------------
# Input : dense TF-IDF vector  (n_features,)
# Layer 1 — FeatureSelector : element-wise multiply by trainable weights w,
#            then Heaviside (straight-through estimator for gradients) →
#            sparse binary mask that selects relevant features.
# Layer 2 — ElementWiseMultiply : input ⊙ selector_output (emphasise
#            selected features while preserving original magnitudes).
# Layer 3 — FC classifier with BatchNorm + Dropout.
#
# The selected feature weights (w after sigmoid) are saved to the report
# so the top discriminative TF-IDF terms are interpretable per task.

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Straight-through Heaviside
# ---------------------------------------------------------------------------

class _HeavisideSTE(torch.autograd.Function):
    """
    Forward  : H(x) = 1 if x >= 0 else 0
    Backward : straight-through — gradient passes as if identity.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def heaviside_ste(x: torch.Tensor) -> torch.Tensor:
    return _HeavisideSTE.apply(x)


# ---------------------------------------------------------------------------
# TransFINN model
# ---------------------------------------------------------------------------

class TransFINN(nn.Module):
    """
    TransFINN classifier.

    Parameters
    ----------
    n_features  : TF-IDF vocabulary size
    n_classes   : number of output classes (1 for binary)
    hidden_dim  : FC hidden layer size
    dropout     : dropout probability in classifier head
    """

    def __init__(
            self,
            n_features: int,
            n_classes: int,
            hidden_dim: int = 256,
            dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        # Feature selector weights — initialised around 0 so ~50% selected at start
        self.feature_weights = nn.Parameter(torch.zeros(n_features))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature selector
        mask = heaviside_ste(self.feature_weights)  # (n_features,)
        selected = x * mask  # broadcast over batch
        # Element-wise emphasis
        emphasised = x * selected  # (B, n_features)
        return self.classifier(emphasised)

    def get_feature_importance(self) -> np.ndarray:
        """Return sigmoid(feature_weights) as a numpy array."""
        with torch.no_grad():
            return torch.sigmoid(self.feature_weights).cpu().numpy()


# ---------------------------------------------------------------------------
# Candidate configs
# ---------------------------------------------------------------------------

def get_transfinn_candidate_configs() -> list[dict]:
    return [
        {"ngram_range": (1, 1), "max_features": 10_000, "min_df": 2,
         "hidden_dim": 256, "dropout": 0.3, "lr": 1e-3, "epochs": 20, "batch_size": 32},
        {"ngram_range": (1, 2), "max_features": 10_000, "min_df": 2,
         "hidden_dim": 256, "dropout": 0.3, "lr": 1e-3, "epochs": 20, "batch_size": 32},
        {"ngram_range": (1, 1), "max_features": 10_000, "min_df": 2,
         "hidden_dim": 512, "dropout": 0.4, "lr": 5e-4, "epochs": 30, "batch_size": 32},
        {"ngram_range": (1, 2), "max_features": 10_000, "min_df": 2,
         "hidden_dim": 512, "dropout": 0.4, "lr": 5e-4, "epochs": 30, "batch_size": 32},
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_dense(X) -> np.ndarray:
    return X.toarray() if issparse(X) else np.asarray(X)


def _build_vectorizer(cfg: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=cfg["ngram_range"],
        max_features=cfg["max_features"],
        min_df=cfg["min_df"],
        sublinear_tf=True,
    )


def _build_label_encoder(series: pd.Series) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(series.dropna())
    return le


def evaluate_binary(y_true, y_pred) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_multiclass(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_task(y_true, y_pred, task_type: str) -> dict:
    return evaluate_binary(y_true, y_pred) if task_type == "binary" else evaluate_multiclass(y_true, y_pred)


def metric_to_optimize(task_type: str) -> str:
    return "f1" if task_type == "binary" else "macro_f1"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one_config(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val_encoded: np.ndarray,
        y_val_labels: list,
        le: LabelEncoder,
        cfg: dict,
        task_type: str,
        device: torch.device,
) -> tuple[TransFINN, dict, dict]:
    """Train TransFINN for a single config, return (model, val_metrics, train_metrics)."""
    torch.manual_seed(RANDOM_STATE)

    n_features = X_train.shape[1]
    n_classes = len(le.classes_)
    out_dim = 1 if (task_type == "binary" and n_classes == 2) else n_classes

    model = TransFINN(
        n_features=n_features,
        n_classes=out_dim,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() if out_dim == 1 else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32 if out_dim == 1 else torch.long)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model.train()
    for epoch in range(cfg["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.squeeze(1) if out_dim == 1 else logits, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        # val
        Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
        logits_val = model(Xv)
        if out_dim == 1:
            pred_val = (torch.sigmoid(logits_val.squeeze(1)) >= 0.5).long().cpu().numpy()
        else:
            pred_val = torch.argmax(logits_val, dim=1).cpu().numpy()
        pred_val_labels = le.inverse_transform(pred_val)
        val_metrics = evaluate_task(y_val_labels, pred_val_labels, task_type)

        # train
        Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
        logits_tr = model(Xt)
        if out_dim == 1:
            pred_tr = (torch.sigmoid(logits_tr.squeeze(1)) >= 0.5).long().cpu().numpy()
        else:
            pred_tr = torch.argmax(logits_tr, dim=1).cpu().numpy()
        pred_tr_labels = le.inverse_transform(pred_tr)
        train_labels = le.inverse_transform(y_train)
        train_metrics = evaluate_task(train_labels, pred_tr_labels, task_type)

    return model, val_metrics, train_metrics


# ---------------------------------------------------------------------------
# Public API — tune
# ---------------------------------------------------------------------------

def tune_transfinn(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_col: str,
        label_col: str,
        task_type: str,
) -> dict:
    device = _get_device()
    logger.info(f"TransFINN device: {device}")

    combined = pd.concat([train_df[label_col], val_df[label_col]], ignore_index=True)
    le = _build_label_encoder(combined)

    y_train_enc = le.transform(train_df[label_col].fillna(""))
    y_val_enc = le.transform(val_df[label_col].fillna(""))
    y_val_labels = val_df[label_col].tolist()

    best_score = -1.0
    best_cfg = None
    best_metrics = None
    best_model = None
    best_vec = None
    search_results: list[dict] = []
    target_metric = metric_to_optimize(task_type)

    for cfg in get_transfinn_candidate_configs():
        vec = _build_vectorizer(cfg)
        X_train = _to_dense(vec.fit_transform(train_df[text_col].fillna("").astype(str)))
        X_val = _to_dense(vec.transform(val_df[text_col].fillna("").astype(str)))

        model, val_metrics, train_metrics = _train_one_config(
            X_train, y_train_enc, X_val, y_val_enc, y_val_labels, le, cfg, task_type, device
        )

        search_results.append({
            "model_name": "transfinn",
            "label_col": label_col,
            "task_type": task_type,
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        })
        logger.info(
            f"  TransFINN cfg {cfg['hidden_dim']}d ngram{cfg['ngram_range']} "
            f"val_{target_metric}={val_metrics[target_metric]:.3f}"
        )

        if val_metrics[target_metric] > best_score:
            best_score = val_metrics[target_metric]
            best_cfg = cfg
            best_metrics = val_metrics
            best_model = model
            best_vec = vec

    if best_cfg is None:
        raise RuntimeError("No valid configuration found during TransFINN tuning.")

    return {
        "best_model": best_model,
        "best_vectorizer": best_vec,
        "best_label_encoder": le,
        "best_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_cfg.items()},
        "best_val_metrics": best_metrics,
        "search_results": search_results,
    }


# ---------------------------------------------------------------------------
# Public API — final fit on train+val, evaluate on test
# ---------------------------------------------------------------------------

def fit_final_transfinn(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_col: str,
        label_col: str,
        task_type: str,
        best_config: dict,
) -> tuple[TransFINN, TfidfVectorizer, LabelEncoder]:
    device = _get_device()
    cfg = {**best_config, "ngram_range": tuple(best_config["ngram_range"])}
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    le = _build_label_encoder(train_val_df[label_col])
    vec = _build_vectorizer(cfg)
    X = _to_dense(vec.fit_transform(train_val_df[text_col].fillna("").astype(str)))
    y = le.transform(train_val_df[label_col].fillna(""))

    n_classes = len(le.classes_)
    out_dim = 1 if (task_type == "binary" and n_classes == 2) else n_classes

    torch.manual_seed(RANDOM_STATE)
    model = TransFINN(
        n_features=X.shape[1],
        n_classes=out_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() if out_dim == 1 else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32 if out_dim == 1 else torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=int(cfg["batch_size"]), shuffle=True)

    model.train()
    for _ in range(int(cfg["epochs"])):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.squeeze(1) if out_dim == 1 else logits, yb)
            loss.backward()
            optimizer.step()

    return model, vec, le


def evaluate_final_transfinn(
        model: TransFINN,
        vectorizer: TfidfVectorizer,
        le: LabelEncoder,
        test_df: pd.DataFrame,
        text_col: str,
        label_col: str,
        task_type: str,
) -> tuple[dict, np.ndarray]:
    device = _get_device()
    X_test = _to_dense(vectorizer.transform(test_df[text_col].fillna("").astype(str)))
    out_dim = model.n_classes

    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(Xt)
        if out_dim == 1:
            pred_ids = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long().cpu().numpy()
        else:
            pred_ids = torch.argmax(logits, dim=1).cpu().numpy()

    y_pred = le.inverse_transform(pred_ids)
    y_true = test_df[label_col].tolist()
    return evaluate_task(y_true, y_pred, task_type), y_pred


def save_transfinn(model, vectorizer, le, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "n_features": model.n_features,
                "n_classes": model.n_classes,
                "hidden_dim": model.classifier[0].out_features,
                "dropout": model.classifier[3].p,
            },
            "vectorizer": vectorizer,
            "label_encoder": le,
        },
        path,
    )


def load_transfinn(path):
    bundle = joblib.load(path)
    cfg = bundle["model_config"]
    model = TransFINN(
        n_features=cfg["n_features"],
        n_classes=cfg["n_classes"],
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, bundle["vectorizer"], bundle["label_encoder"]


def get_top_features(
        model: TransFINN,
        vectorizer: TfidfVectorizer,
        top_n: int = 30,
) -> list[dict]:
    """Return the top_n most important TF-IDF features by selector weight."""
    importance = model.get_feature_importance()
    vocab = vectorizer.get_feature_names_out()
    indices = np.argsort(importance)[::-1][:top_n]
    return [{"feature": vocab[i], "weight": float(importance[i])} for i in indices]

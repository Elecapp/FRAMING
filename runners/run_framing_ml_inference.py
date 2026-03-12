# runners/run_framing_ml_inference.py
# Apply best ML framing models to unseen data (LOC inference set).
#
# Reads final_model_selection.json produced by run_framing_ml_models.py
# to determine which model to load per task.
#
# Input  : data/processed/LOC_v2_event_detection.csv
#          data/processed/LOC_v2_preprocessed.csv   (for text_ml / text_linguistic)
#          — LOC_v2_event_detection filtered to pred_event=1 and FAS IDs removed
#
# Output : data/processed/LOC_v2_framing_ml.csv
#          reports/framing_ml/ml_inference_metadata.json

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.framing_ml.transfinn_models import TransFINN, load_transfinn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
LOC_PATH = ROOT / "data" / "processed" / "LOC_v2_event_detection.csv"
LOC_PREPROCESSED_PATH = ROOT / "data" / "processed" / "LOC_v2_preprocessed.csv"
FAS_PATH = ROOT / "data" / "processed" / "FAS_preprocessed.csv"
OUTPUT = ROOT / "data" / "processed" / "LOC_v2_framing_ml.csv"
MODELS_DIR = ROOT / "models" / "framing_ml"
REPORTS_DIR = ROOT / "reports" / "framing_ml"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

TEXT_ML = "text_ml"            # for TF-IDF models (sklearn, transfinn, xgboost)
TEXT_LINGUISTIC = "text_linguistic"  # for BERT
ID_COL = "id"

TASK_CONFIG: dict[str, dict] = {
    "victim_blaming": {
        "out_col": "victim_blaming_ml",
    },
    "perpetrator_justification": {
        "out_col": "perp_justified_ml",
    },
    "tone": {
        "out_col": "tone_ml",
    },
}

DEFAULT_FINAL_MODELS: dict[str, str] = {
    "victim_blaming": "logreg",
    "perpetrator_justification": "linear_svm",
    "tone": "logreg",
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_sklearn_model(task_name: str, model_name: str):
    path = MODELS_DIR / f"{task_name}__{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def _load_xgb_bundle(task_name: str) -> dict:
    path = MODELS_DIR / f"{task_name}__xgboost.joblib"
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {path}")
    return joblib.load(path)


def _load_transfinn(task_name: str) -> tuple:
    path = MODELS_DIR / f"{task_name}__transfinn.joblib"
    if not path.exists():
        raise FileNotFoundError(f"TransFINN model not found: {path}")
    return load_transfinn(path)


def _load_bert(task_name: str) -> tuple:
    model_dir = MODELS_DIR / f"{task_name}__bert" / "final_model"
    if not model_dir.exists():
        raise FileNotFoundError(f"BERT model not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    return model, tokenizer


# ---------------------------------------------------------------------------
# Predict functions
# ---------------------------------------------------------------------------

def _predict_sklearn(model, texts: pd.Series) -> np.ndarray:
    return model.predict(texts.fillna("").astype(str))


def _predict_xgb(bundle: dict, texts: pd.Series, task_type: str) -> np.ndarray:
    model = bundle["model"]
    id2label = bundle.get("id2label")
    raw = model.predict(texts.fillna("").astype(str))
    if task_type == "multiclass" and id2label:
        return np.array([id2label[int(x)] for x in raw])
    return raw


def _predict_transfinn(
        model: TransFINN, vectorizer, le, texts: pd.Series
) -> np.ndarray:
    from scipy.sparse import issparse
    device = _get_device()
    X = vectorizer.transform(texts.fillna("").astype(str))
    X = X.toarray() if issparse(X) else np.asarray(X)

    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(Xt)
        if model.n_classes == 1:
            pred_ids = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long().cpu().numpy()
        else:
            pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
    return le.inverse_transform(pred_ids)


def _predict_bert(
        model, tokenizer, texts: pd.Series, batch_size: int = 8, max_length: int = 256
) -> np.ndarray:
    device = _get_device()
    model.to(device)
    model.eval()

    text_list = texts.fillna("").astype(str).tolist()
    id2label = model.config.id2label
    all_preds: list = []

    for start in range(0, len(text_list), batch_size):
        batch = text_list[start:start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
        for pid in pred_ids:
            key = pid if pid in id2label else str(pid)
            all_preds.append(id2label[key])

    return np.array(all_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for p in (LOC_PATH, LOC_PREPROCESSED_PATH, FAS_PATH):
        if not p.exists():
            logger.critical(f"Required file not found: {p}")
            sys.exit(1)

    # Load best model selection
    selection_path = REPORTS_DIR / "final_model_selection.json"
    if selection_path.exists():
        with open(selection_path, encoding="utf-8") as fh:
            final_models: dict[str, str] = json.load(fh)
        logger.info(f"Loaded model selection: {final_models}")
    else:
        final_models = DEFAULT_FINAL_MODELS
        logger.warning(
            f"final_model_selection.json not found — using defaults: {final_models}"
        )

    # Build inference set: LOC pred_event=1 minus FAS IDs
    fas_ids = set(pd.read_csv(FAS_PATH, usecols=["id"], keep_default_na=False)["id"])
    loc = pd.read_csv(LOC_PATH, keep_default_na=False)

    if "pred_event" not in loc.columns:
        logger.critical("LOC file has no pred_event column — run run_event_detection.py first.")
        sys.exit(1)

    n_total = len(loc)
    loc = loc[loc["pred_event"] == 1].copy()
    n_positive = len(loc)
    overlap = len(fas_ids & set(loc[ID_COL]))
    df = loc[~loc[ID_COL].isin(fas_ids)].reset_index(drop=True)
    logger.info(
        f"LOC: {n_total} total → {n_positive} pred_event=1 → "
        f"{overlap} FAS overlap removed → {len(df)} for inference"
    )

    # Merge preprocessed text columns (text_ml, text_linguistic)
    loc_pre = pd.read_csv(
        LOC_PREPROCESSED_PATH,
        usecols=[ID_COL, TEXT_ML, TEXT_LINGUISTIC],
        keep_default_na=False,
    )
    before = len(df)
    df = df.merge(loc_pre, on=ID_COL, how="left")
    missing = df[TEXT_ML].isna().sum()
    if missing > 0:
        logger.warning(
            f"{missing}/{before} articles have no preprocessed text — "
            f"they will be predicted on empty string."
        )

    out = df.copy()

    for task_name, task_info in TASK_CONFIG.items():
        model_name = final_models.get(task_name, DEFAULT_FINAL_MODELS[task_name])
        out_col = task_info["out_col"]
        logger.info(f"  {task_name} → {model_name}")

        if model_name in ("logreg", "linear_svm", "naive_bayes"):
            model = _load_sklearn_model(task_name, model_name)
            preds = _predict_sklearn(model, df[TEXT_ML])

        elif model_name == "xgboost":
            bundle = _load_xgb_bundle(task_name)
            task_type = "multiclass" if task_name == "tone" else "binary"
            preds = _predict_xgb(bundle, df[TEXT_ML], task_type)

        elif model_name == "transfinn":
            tfmodel, vec, le = _load_transfinn(task_name)
            preds = _predict_transfinn(tfmodel, vec, le, df[TEXT_ML])

        elif model_name == "bert":
            model, tokenizer = _load_bert(task_name)
            preds = _predict_bert(model, tokenizer, df[TEXT_LINGUISTIC])

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        out[out_col] = preds

    out.to_csv(OUTPUT, index=False, encoding="utf-8")
    logger.info(f"Predictions saved → {OUTPUT}")

    metadata = {
        "loc_path": str(LOC_PATH),
        "loc_preprocessed_path": str(LOC_PREPROCESSED_PATH),
        "output_path": str(OUTPUT),
        "n_loc_total": n_total,
        "n_loc_positive": n_positive,
        "n_fas_overlap_removed": overlap,
        "n_inference": len(df),
        "final_models": final_models,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "ml_inference_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    print(
        f"\n✓  Inference complete\n"
        f"   Articles : {len(df)}\n"
        f"   Models   : {final_models}\n"
        f"   Output   : {OUTPUT}\n"
        f"   (FAS overlap removed: {overlap})\n"
    )


if __name__ == "__main__":
    main()
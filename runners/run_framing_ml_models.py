# runners/run_framing_ml_models.py
# Train, tune, and evaluate all ML framing models on FAS splits.
#
# Models: logreg, linear_svm, naive_bayes, xgboost, transfinn, bert
# Tasks : victim_blaming (binary), perpetrator_justification (binary), tone (multiclass)
#
# Input  : data/processed/ml_splits/{fas_train, fas_val, fas_test}.csv
#          — produced by prepare_framing_ml_splits.py
#
# TF-IDF models (logreg, svm, nb, xgboost, transfinn) use: text_ml
# BERT uses: text_linguistic  (tokenizer handles lowercasing / punctuation)
#
# Outputs (reports/framing_ml/):
#   {task}__{model}.json           — per-model report (params + val + test metrics)
#   test_predictions_all_models.csv — y_true / y_pred for all models × tasks
#   ml_models_leaderboard.csv      — compact comparison table
#   ml_models_summary.txt          — human-readable summary
#   final_model_selection.json     — best model per task (for inference)
#   transfinn_top_features.json    — top TF-IDF features selected by TransFINN
#
# Models saved to models/framing_ml/

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.framing_ml.baselines import (
    RANDOM_STATE,
    evaluate_final_model,
    fit_final_model,
    tune_baseline,
)
from src.framing_ml.transformer_models import (
    evaluate_final_transformer,
    fit_final_transformer,
    tune_transformer,
)
from src.framing_ml.transfinn_models import (
    evaluate_final_transfinn,
    fit_final_transfinn,
    get_top_features,
    save_transfinn,
    tune_transfinn,
)
from src.framing_ml.xgb_models import (
    evaluate_final_xgb,
    fit_final_xgb,
    tune_xgb,
)

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
SPLIT_DIR = ROOT / "data" / "processed" / "ml_splits"
MODELS_DIR = ROOT / "models" / "framing_ml"
REPORTS_DIR = ROOT / "reports" / "framing_ml"

TRAIN_PATH = SPLIT_DIR / "fas_train.csv"
VAL_PATH = SPLIT_DIR / "fas_val.csv"
TEST_PATH = SPLIT_DIR / "fas_test.csv"

# ---------------------------------------------------------------------------
# Tasks and models
# ---------------------------------------------------------------------------

# TF-IDF models use text_ml (pre-cleaned, lowercased)
# BERT uses text_linguistic (richer, tokenizer handles the rest)
TEXT_ML = "text_ml"
TEXT_LINGUISTIC = "text_linguistic"
ID_COL = "id"

TASKS: dict[str, dict] = {
    "victim_blaming": {
        "label_col": "victim_blaming_gt",
        "task_type": "binary",
    },
    "perpetrator_justification": {
        "label_col": "perp_justified_gt",
        "task_type": "binary",
    },
    "tone": {
        "label_col": "tone_label_gt",
        "task_type": "multiclass",
    },
}

# Models that use TF-IDF sparse features
TFIDF_MODELS = ("logreg", "linear_svm", "naive_bayes", "xgboost", "transfinn")
# Models that use contextualised text
TRANSFORMER_MODELS = ("bert",)

ALL_MODEL_NAMES = list(TFIDF_MODELS) + list(TRANSFORMER_MODELS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def metric_to_optimize(task_type: str) -> str:
    return "f1" if task_type == "binary" else "macro_f1"


# ---------------------------------------------------------------------------
# Per-model runners
# ---------------------------------------------------------------------------

def run_sparse_baseline(
        train_df, val_df, test_df, task_name, label_col, task_type, model_name
) -> tuple[dict, pd.DataFrame]:
    tuning = tune_baseline(
        train_df=train_df, val_df=val_df,
        text_col=TEXT_ML, label_col=label_col,
        model_name=model_name, task_type=task_type,
    )
    final = fit_final_model(
        train_df=train_df, val_df=val_df,
        text_col=TEXT_ML, label_col=label_col,
        model_name=model_name,
        best_config=tuning["best_config"],
        task_type=task_type,
    )
    model_path = MODELS_DIR / f"{task_name}__{model_name}.joblib"
    joblib.dump(final, model_path)

    test_metrics, test_pred = evaluate_final_model(
        model=final, test_df=test_df,
        text_col=TEXT_ML, label_col=label_col, task_type=task_type,
    )
    report = {
        "task_name": task_name, "label_col": label_col, "task_type": task_type,
        "model_name": model_name, "text_col": TEXT_ML,
        "best_config": tuning["best_config"],
        "val_metrics": tuning["best_val_metrics"],
        "test_metrics": test_metrics,
        "search_results": tuning["search_results"],
        "model_path": str(model_path),
    }
    pred_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].tolist(),
        "task_name": task_name, "model_name": model_name,
        "y_true": test_df[label_col].tolist(),
        "y_pred": list(test_pred),
    })
    return report, pred_df


def run_xgboost(
        train_df, val_df, test_df, task_name, label_col, task_type
) -> tuple[dict, pd.DataFrame]:
    tuning = tune_xgb(
        train_df=train_df, val_df=val_df,
        text_col=TEXT_ML, label_col=label_col, task_type=task_type,
    )
    final, label2id, id2label = fit_final_xgb(
        train_df=train_df, val_df=val_df,
        text_col=TEXT_ML, label_col=label_col,
        task_type=task_type, best_config=tuning["best_config"],
    )
    model_path = MODELS_DIR / f"{task_name}__xgboost.joblib"
    joblib.dump({"model": final, "label2id": label2id, "id2label": id2label}, model_path)

    test_metrics, test_pred = evaluate_final_xgb(
        model=final, test_df=test_df,
        text_col=TEXT_ML, label_col=label_col,
        task_type=task_type, id2label=id2label,
    )
    report = {
        "task_name": task_name, "label_col": label_col, "task_type": task_type,
        "model_name": "xgboost", "text_col": TEXT_ML,
        "best_config": tuning["best_config"],
        "val_metrics": tuning["best_val_metrics"],
        "test_metrics": test_metrics,
        "search_results": tuning["search_results"],
        "model_path": str(model_path),
    }
    pred_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].tolist(),
        "task_name": task_name, "model_name": "xgboost",
        "y_true": test_df[label_col].tolist(),
        "y_pred": list(test_pred),
    })
    return report, pred_df


def run_transfinn(
        train_df, val_df, test_df, task_name, label_col, task_type
) -> tuple[dict, pd.DataFrame]:
    tuning = tune_transfinn(
        train_df=train_df, val_df=val_df,
        text_col=TEXT_ML, label_col=label_col, task_type=task_type,
    )
    final_model, final_vec, final_le = fit_final_transfinn(
        train_df=train_df, val_df=val_df,
        text_col=TEXT_ML, label_col=label_col,
        task_type=task_type, best_config=tuning["best_config"],
    )
    model_path = MODELS_DIR / f"{task_name}__transfinn.joblib"
    save_transfinn(final_model, final_vec, final_le, model_path)

    # Top features — interpretability output for the paper
    top_features = get_top_features(final_model, final_vec, top_n=30)
    save_json(
        {"task": task_name, "top_features": top_features},
        REPORTS_DIR / f"{task_name}__transfinn_top_features.json",
    )

    test_metrics, test_pred = evaluate_final_transfinn(
        model=final_model, vectorizer=final_vec, le=final_le,
        test_df=test_df, text_col=TEXT_ML,
        label_col=label_col, task_type=task_type,
    )
    report = {
        "task_name": task_name, "label_col": label_col, "task_type": task_type,
        "model_name": "transfinn", "text_col": TEXT_ML,
        "best_config": tuning["best_config"],
        "val_metrics": tuning["best_val_metrics"],
        "test_metrics": test_metrics,
        "search_results": tuning["search_results"],
        "model_path": str(model_path),
    }
    pred_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].tolist(),
        "task_name": task_name, "model_name": "transfinn",
        "y_true": test_df[label_col].tolist(),
        "y_pred": list(test_pred),
    })
    return report, pred_df


def run_bert(
        train_df, val_df, test_df, task_name, label_col, task_type
) -> tuple[dict, pd.DataFrame]:
    bert_dir = MODELS_DIR / f"{task_name}__bert"
    bert_dir.mkdir(parents=True, exist_ok=True)

    tuning = tune_transformer(
        train_df=train_df, val_df=val_df,
        text_col=TEXT_LINGUISTIC, label_col=label_col,
        task_type=task_type, run_dir=bert_dir / "tuning",
    )
    final_dir, tokenizer = fit_final_transformer(
        train_df=train_df, val_df=val_df,
        text_col=TEXT_LINGUISTIC, label_col=label_col,
        task_type=task_type, best_config=tuning["best_config"],
        label2id=tuning["label2id"], id2label=tuning["id2label"],
        run_dir=bert_dir,
    )
    test_metrics, test_pred = evaluate_final_transformer(
        model_dir=final_dir, tokenizer=tokenizer,
        test_df=test_df, text_col=TEXT_LINGUISTIC,
        label_col=label_col, task_type=task_type,
        label2id=tuning["label2id"],
    )
    report = {
        "task_name": task_name, "label_col": label_col, "task_type": task_type,
        "model_name": "bert", "text_col": TEXT_LINGUISTIC,
        "best_config": tuning["best_config"],
        "val_metrics": tuning["best_val_metrics"],
        "test_metrics": test_metrics,
        "search_results": tuning["search_results"],
        "model_path": str(final_dir),
    }
    pred_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].tolist(),
        "task_name": task_name, "model_name": "bert",
        "y_true": test_df[label_col].tolist(),
        "y_pred": list(test_pred),
    })
    return report, pred_df


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def build_summary_txt(all_results: list[dict]) -> str:
    sep = "=" * 72
    sep2 = "-" * 72
    lines = [
        sep,
        "FRAMING ML — TRAINING SUMMARY",
        f"Random seed: {RANDOM_STATE}",
        f"Models: {', '.join(ALL_MODEL_NAMES)}",
        sep,
    ]
    for task_name in TASKS:
        lines += [sep2, task_name.upper(), sep2]
        task_results = [r for r in all_results if r["task_name"] == task_name]
        for r in task_results:
            vm = r["val_metrics"]
            tm = r["test_metrics"]
            lines.append(f"  {r['model_name']:<16}")
            lines.append(f"    val  : {vm}")
            lines.append(f"    test : {tm}")
            lines.append(f"    cfg  : {r['best_config']}")
            lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def build_leaderboard(all_results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in all_results:
        tm = r["test_metrics"]
        vm = r["val_metrics"]
        target = metric_to_optimize(r["task_type"])
        rows.append({
            "task": r["task_name"],
            "model": r["model_name"],
            "val_" + target: round(vm.get(target, float("nan")), 4),
            "test_" + target: round(tm.get(target, float("nan")), 4),
            **{f"test_{k}": round(v, 4) for k, v in tm.items() if k != target},
        })
    return pd.DataFrame(rows)


def select_best_models(all_results: list[dict]) -> dict[str, str]:
    """Return {task_name: best_model_name} based on test metric."""
    best: dict[str, tuple[str, float]] = {}
    for r in all_results:
        task = r["task_name"]
        target = metric_to_optimize(r["task_type"])
        score = r["test_metrics"].get(target, -1.0)
        if task not in best or score > best[task][1]:
            best[task] = (r["model_name"], score)
    return {task: model for task, (model, _) in best.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for p in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not p.exists():
            logger.critical(f"Split file not found: {p}  — run prepare_framing_ml_splits.py first")
            sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    val_df = pd.read_csv(VAL_PATH, keep_default_na=False)
    test_df = pd.read_csv(TEST_PATH, keep_default_na=False)

    logger.info(f"Splits — train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    all_results: list[dict] = []
    all_predictions: list[pd.DataFrame] = []

    for task_name, task_cfg in TASKS.items():
        label_col = task_cfg["label_col"]
        task_type = task_cfg["task_type"]

        for model_name in ALL_MODEL_NAMES:
            logger.info(f"▶  {task_name} | {model_name}")

            if model_name in ("logreg", "linear_svm", "naive_bayes"):
                report, pred_df = run_sparse_baseline(
                    train_df, val_df, test_df, task_name, label_col, task_type, model_name
                )
            elif model_name == "xgboost":
                report, pred_df = run_xgboost(
                    train_df, val_df, test_df, task_name, label_col, task_type
                )
            elif model_name == "transfinn":
                report, pred_df = run_transfinn(
                    train_df, val_df, test_df, task_name, label_col, task_type
                )
            elif model_name == "bert":
                report, pred_df = run_bert(
                    train_df, val_df, test_df, task_name, label_col, task_type
                )
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

            target = metric_to_optimize(task_type)
            logger.info(
                f"   val_{target}={report['val_metrics'].get(target, '?'):.3f}  "
                f"test_{target}={report['test_metrics'].get(target, '?'):.3f}"
            )

            save_json(report, REPORTS_DIR / f"{task_name}__{model_name}.json")
            all_results.append(report)
            all_predictions.append(pred_df)

    # Aggregate outputs
    pd.concat(all_predictions, ignore_index=True).to_csv(
        REPORTS_DIR / "test_predictions_all_models.csv", index=False, encoding="utf-8"
    )

    (REPORTS_DIR / "ml_models_summary.txt").write_text(
        build_summary_txt(all_results), encoding="utf-8"
    )

    build_leaderboard(all_results).to_csv(
        REPORTS_DIR / "ml_models_leaderboard.csv", index=False, encoding="utf-8"
    )

    best_models = select_best_models(all_results)
    save_json(best_models, REPORTS_DIR / "final_model_selection.json")

    logger.info(f"Best models per task: {best_models}")
    logger.info(f"Models saved  : {MODELS_DIR}")
    logger.info(f"Reports saved : {REPORTS_DIR}")

    print(
        f"\n-  Training complete\n"
        f"   Best models: {best_models}\n"
        f"   Reports: {REPORTS_DIR}\n"
    )


if __name__ == "__main__":
    main()

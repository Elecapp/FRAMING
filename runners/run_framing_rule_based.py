# runners/run_framing_rule_based.py
# Apply tuned rule-based framing detectors to any unseen preprocessed dataset.
#
# Reads best params from config/framing_params/ (produced by
# run_rule_based_param_tuning.py on FAS) and applies all framing dimensions.
#
# Do NOT run on FAS — params were estimated on that data.
# Intended for: LOC_v2_event_detection.csv or any future unseen corpus.

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# Dataset to annotate — must be unseen during parameter tuning
INPUT = ROOT / "data" / "processed" / "LOC_v2_event_detection.csv"

PARAMS_DIR = ROOT / "config" / "framing_params"
OUTPUT_DIR = ROOT / "data" / "processed"

TEXT_COL = "text_linguistic"
BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Imports from package
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT))

from src.framing_rule_based.parsing import build_rule_based_views
from src.framing_rule_based.victim_blaming import apply_victim_blaming
from src.framing_rule_based.justification import apply_justification
from src.framing_rule_based.tone import apply_tone
from src.framing_rule_based.roles_focus import apply_roles_focus


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not INPUT.exists():
        logger.critical(f"Input file not found: {INPUT}")
        sys.exit(1)

    if not PARAMS_DIR.exists():
        logger.critical(
            f"Params directory not found: {PARAMS_DIR}\n"
            "Run run_rule_based_param_tuning.py first."
        )
        sys.exit(1)

    logger.info(f"Loading: {INPUT.name}")
    df = pd.read_csv(INPUT, keep_default_na=False)
    logger.info(f"Loaded {len(df)} articles")

    # Filter to crime-related articles only when pred_event column is present
    # (i.e. when running on LOC output from event detection)
    if "pred_event" in df.columns:
        n_total = len(df)
        df = df[df["pred_event"] == 1].copy()
        logger.info(
            f"Filtered to pred_event=1: {len(df)} / {n_total} articles "
            f"({len(df) / n_total:.1%})"
        )

    if TEXT_COL not in df.columns:
        logger.critical(
            f"Column '{TEXT_COL}' not found. "
            f"Available: {list(df.columns)}"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Build linguistic views (regex / spaCy / Stanza)
    # ------------------------------------------------------------------
    logger.info("Building linguistic views...")
    views = build_rule_based_views(
        df[TEXT_COL],
        batch_size=BATCH_SIZE,
        show_progress=True,
    )
    logger.info(
        f"Views built — "
        f"regex: {len(views['regex'])}  "
        f"spaCy: {len(views['spacy'])}  "
        f"Stanza: {len(views['stanza'])}"
    )

    # ------------------------------------------------------------------
    # 3. Apply framing detectors
    # ------------------------------------------------------------------
    logger.info("Applying victim-blaming detector...")
    df = apply_victim_blaming(df, views, PARAMS_DIR)

    logger.info("Applying perpetrator-justification detector...")
    df = apply_justification(df, views, PARAMS_DIR)

    logger.info("Applying tone detector...")
    df = apply_tone(df, views, PARAMS_DIR)

    logger.info("Applying victim agency and narrative focus...")
    df = apply_roles_focus(df, views)

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{INPUT.stem}_framing.csv"

    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved : {output_path}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    n = len(df)

    def _pct(col):
        return f"{df[col].sum()} ({df[col].mean():.1%})"

    def _dist(col, labels):
        vc = df[col].value_counts()
        return "  ".join(f"{lbl}: {vc.get(lbl, 0)} ({vc.get(lbl, 0) / n:.1%})" for lbl in labels)

    print(f"\n{'=' * 60}")
    print(f"FRAMING DETECTION COMPLETE  —  {n} articles")
    print(f"{'=' * 60}")
    print(f"  Victim blaming          : {_pct('victim_blaming_rulebased')}")
    print(f"  Perpetrator justification: {_pct('perp_justified_rulebased')}")
    print(f"  Tone  : {_dist('tone_rulebased', ['neutro', 'emotivo', 'sensazionalistico'])}")
    print(f"  Agency: {_dist('victim_agency_class', ['alta', 'media', 'bassa'])}")
    print(f"  Focus : {_dist('focus_rulebased', ['vittima', 'carnefice', 'bilanciato'])}")
    print(f"{'=' * 60}")
    print(f"  Output: {output_path}\n")


if __name__ == "__main__":
    main()

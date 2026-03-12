# FRAMING — Femicide Responsibility Automated MonitorING

End-to-end NLP pipeline for the automated detection and framing analysis of femicide reporting in Italian news.

## Overview

FRAMING integrates four components into a unified framework:

1. **Event detection** — filters a raw news corpus to femicide-relevant articles
2. **Rule-based framing** — lexicon-grounded detection of victim blaming, perpetrator justification, and tone
3. **Supervised framing classification** — TF-IDF-based classifiers trained on the Framing Annotation Subset (FAS)
4. **Syntactic framing indicators** — annotation-free indicators of victim agency and narrative focus derived from dependency parsing

## Repository Structure
```
FRAMING/
├── config/
│   └── framing_params/         # Tuned rule-based hyperparameters
├── data/
│   ├── raw/                    # Raw and annotated input data
│   └── processed/              # Preprocessed article CSVs
├── models/
│   ├── event_detection/        # Saved event detection models
│   └── framing_ml/             # Saved supervised framing models
├── reports/
│   ├── event_detection/
│   ├── framing_dimensions/
│   ├── framing_indicators/
│   ├── framing_ml/
│   ├── paper_numbers/
│   └── rule_based_tuning/
├── src/
│   ├── event_detection/
│   ├── framing_ml/
│   ├── framing_rule_based/
│   ├── ground_truth/
│   ├── preprocessing/
│   └── utils/
├── runners/                    # Executable pipeline scripts (see below)
└── FRAMING-Annotation Guidelines.pdf
```

## Pipeline Execution Order

Run the scripts in `runners/` in the following order:
```bash
# 1. Build ground truth from raw annotations
python runners/run_build_ground_truth.py

# 2. Preprocess articles
python runners/run_text_preprocess.py

# 3. Event detection — train, evaluate, and apply to corpus
python runners/run_event_detection.py

# 4. Rule-based framing — hyperparameter tuning on FAS
python runners/run_rule_based_param_tuning.py

# 5. Rule-based framing — apply to LOC
python runners/run_framing_rule_based.py

# 6. Syntactic framing indicators — compute and analyse
python runners/analyze_framing_dimensions.py
python runners/analyze_framing_indicators.py

# 7. Supervised framing — prepare splits
python runners/prepare_framing_ml_splits.py

# 8. Supervised framing — train and evaluate models
python runners/run_framing_ml_models.py

# 9. Supervised framing — corpus-level inference
python runners/run_framing_ml_inference.py
```

Reporting scripts can be run at any point after the corresponding pipeline stage:
```bash
python runners/report_datasets.py
python runners/report_ground_truth.py
python runners/report_event_detection.py
```

## Datasets

The Event Detection Subset (EDS) and the Framing Annotation Subset (FAS) are released together with annotation guidelines designed to support reproducible and ML-friendly framing annotation. The guidelines are freely accessible in this repository (`FRAMING-Annotation Guidelines.pdf`); the datasets are available upon request due to copyright restrictions on the source articles.

## Citation

If you use FRAMING or the associated resources, please cite:
```bibtex
@inproceedings{framing2026,
  title     = {FRAMING: Femicide Responsibility Automated MonitorING},
  booktitle = {Proceedings of ECML PKDD},
  year      = {2026}
}
```

## License

Code: MIT License.  
Annotation guidelines: CC BY 4.0.  
Datasets: available upon request, subject to non-disclosure agreement.

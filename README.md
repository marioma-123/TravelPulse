# TravelPulse - Prediction API (deployment)

FastAPI service that predicts the nightly price (USD) of an **existing** short-term rental listing
(Amsterdam, Barcelona, Berlin, Columbus) with the modeling team's XGBoost model.

The model was trained on a hand-built feature table (not a scikit-learn Pipeline), so `app/features.py`
re-creates the notebook's preprocessing for one listing. `tests/test_parity.py` proves it matches the notebook.

## Layout

```text
deployment/
├── app/
│   ├── main.py           endpoints (/health, /predict) + model loading at startup
│   ├── schemas.py        input/output format and validation
│   ├── features.py       raw listing -> the model's feature row (copy of the notebook logic)
│   ├── model_loader.py   loads xgb_model.json + artifacts.json + landmarks; runs the prediction
│   └── config.py         model folder, city centres
├── models/
│   ├── make_dummy_model.py   builds a tiny fake bundle in models/dummy/ for development
│   └── (real files go here: xgb_model.json, artifacts.json, landmarks_clean.csv, parity_sample.csv)
├── export/export_for_deployment.py   cell to paste at the end of the modeling notebook
├── tests/                automated tests
├── Dockerfile / .dockerignore
├── requirements.txt      runtime libraries
└── requirements-dev.txt  runtime + test libraries
```

## Run with the dummy model (from inside `deployment/`)

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m models.make_dummy_model                      # creates models/dummy/
MODEL_DIR=models/dummy uvicorn app.main:app --reload   # http://127.0.0.1:8000/docs
python -m pytest
```

## Switch to the real model

1. In the modeling notebook, paste `export/export_for_deployment.py` as the last cell and run it.
   It creates `export_for_deployment/` with `xgb_model.json`, `artifacts.json`, `parity_sample.csv`.
2. Copy those 3 files + `data/processed/landmarks_clean.csv` into `deployment/models/`.
3. `python -m pytest` - `test_parity.py` now runs and must pass.
4. `uvicorn app.main:app --reload` (no `MODEL_DIR` needed).

Re-export whenever the modeling team retrains or changes the features. Landmarks must be the same file
used for training.

## Docker / Podman (real model files must be in models/ first)

```bash
docker build -t travelpulse-api .
docker run --rm -p 8000:8000 travelpulse-api
```

## Open decisions

- `as_of_date` default: the data snapshot date (`reference_date` in artifacts.json), not today.
- Review scores are validated on a 0-5 scale (seen in the real parity sample; re-check if the data changes).
- Pin runtime library versions (`pip freeze` in a clean venv) and match the modeling team's xgboost version.
- Where the model files live for hosting (committed to Git vs delivered separately).

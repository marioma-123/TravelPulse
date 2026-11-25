"""بيضيف قيم النطاق (conformal) لـ artifacts.json من نتائج الـ validation.

في الـ notebook (بعد التدريب) صدّر ملف CSV من عمودين على مجموعة الـ validation
(الوحدات اللي الموديل ماتدربش عليها):
    y_true_log : log1p(السعر الحقيقي)
    y_pred_log : توقع الموديل (برضه في الـ log)

    python -m tools.add_conformal models/calibration.csv        (من جوه deployment/)
"""
import json
import sys

import pandas as pd

from app.config import ARTIFACTS_FILE, get_model_dir
from app.interval import compute_conformal_quantiles


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    result = compute_conformal_quantiles(df["y_true_log"], df["y_pred_log"])
    path = get_model_dir() / ARTIFACTS_FILE
    artifacts = json.loads(path.read_text())
    artifacts["conformal"] = result
    path.write_text(json.dumps(artifacts))
    print("saved:", result)


if __name__ == "__main__":
    main(sys.argv[1])

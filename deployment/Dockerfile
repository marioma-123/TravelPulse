FROM docker.io/library/python:3.15-slim

# لا نكتب ملفات .pyc ونطبع اللوج فوراً (أسهل في قراءة اللوج على السحابة)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# نسخ requirements لوحده الأول: لو الكود اتغير والمكتبات لأ، Docker يعيد استخدام الطبقة دي
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# ملفات الموديل: xgb_model.json + artifacts.json + landmarks_clean.csv
COPY models ./models
# ملفات التحليلات (mispricing / market): الـ endpoints بترجع 503 لو مش موجودة
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

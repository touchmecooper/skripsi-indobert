# Mengikuti pola resmi HF: jalankan sebagai user non-root (UID 1000).
#
# CATATAN: Dockerfile ini TIDAK meng-copy folder model_output/ (model
# ~500MB) karena model di-download otomatis dari Hugging Face Hub saat
# container start (lihat HF_MODEL_REPO & HF_TOKEN di app.py / Space secrets).
# Ini bikin image jauh lebih ringan & build lebih cepat.

FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies dulu (Docker layer caching lebih efisien)
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy source code (TIDAK termasuk model_output/, lihat .dockerignore)
COPY --chown=user . $HOME/app

EXPOSE 8080

CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-8080}"]

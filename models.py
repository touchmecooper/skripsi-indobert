"""
models.py
---------
Definisi struktur database untuk menyimpan artikel yang diupload
beserta hasil klasifikasinya.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Article(db.Model):
    __tablename__ = "articles"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)          # teks hasil ekstraksi PDF

    # Referensi ke file PDF asli yang disimpan lokal
    original_filename = db.Column(db.String(255), nullable=False)  # nama file saat diupload
    stored_filename = db.Column(db.String(255), nullable=False)    # nama file unik di disk
    page_count = db.Column(db.Integer, nullable=True)

    # Hasil klasifikasi model
    category = db.Column(db.String(50), nullable=False)       # Teknologi/Medis/Edukasi
    confidence = db.Column(db.Float, nullable=False)          # 0-1
    prob_teknologi = db.Column(db.Float, nullable=False)
    prob_medis = db.Column(db.Float, nullable=False)
    prob_edukasi = db.Column(db.Float, nullable=False)

    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "confidence": self.confidence,
            "probabilities": {
                "Teknologi": self.prob_teknologi,
                "Medis": self.prob_medis,
                "Edukasi": self.prob_edukasi,
            },
            "uploaded_at": self.uploaded_at.strftime("%Y-%m-%d %H:%M"),
        }

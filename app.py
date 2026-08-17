"""
app.py
------
Aplikasi Flask untuk testing DAN penggunaan model IndoBERT hasil fine-tuning.

Fungsi:
1. Memuat model dan tokenizer dari folder `model_output/` (hasil save_pretrained).
2. Menyediakan endpoint web (form) dan endpoint API (/predict) untuk melakukan
   inferensi klasifikasi teks artikel ke dalam 3 kategori:
   teknologi (0), medis (1), edukasi (2).
3. Menampilkan kategori prediksi beserta confidence score per kelas.
4. Menyediakan fitur UPLOAD artikel (tempel teks atau file .txt) yang otomatis
   diklasifikasi dan disimpan ke database (SQLite).
5. Menyediakan halaman daftar & pencarian artikel tersimpan, dikelompokkan
   per kategori.

Cara menjalankan:
    pip install -r requirements.txt
    python app.py

Lalu buka browser ke http://127.0.0.1:5000
"""

import os
import re
import uuid
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models import db, Article

# ----------------------------------------------------------------------
# KONFIGURASI MODEL HUGGING FACE
# ----------------------------------------------------------------------

HF_MODEL_REPO = os.getenv(
    "MODEL_REPO_ID",
    "cooperss/skripsi-indobert-model"
)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN belum diset. "
        "Set environment variable HF_TOKEN "
        "untuk mengakses model private di Hugging Face."
    )

# ----------------------------------------------------------------------
# KONFIGURASI APLIKASI
# ----------------------------------------------------------------------

ID2LABEL = {
    0: "Teknologi",
    1: "Medis",
    2: "Edukasi",
}

# Harus sama dengan konfigurasi saat training
MAX_LENGTH = 512

# Gunakan GPU jika tersedia, jika tidak gunakan CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Database SQLite
DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "articles.db"
)

# Folder penyimpanan PDF
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "uploaded_pdfs"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# Format file yang diperbolehkan
ALLOWED_EXTENSIONS = {"pdf"}

# Batas ukuran upload: 16 MB
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# ----------------------------------------------------------------------
# LOAD MODEL & TOKENIZER
# ----------------------------------------------------------------------

def load_model_and_tokenizer():
    print(f"[INFO] Memuat model dari Hugging Face: {HF_MODEL_REPO}")

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_REPO,
        token=HF_TOKEN,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_REPO,
        token=HF_TOKEN,
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"[INFO] Model berhasil dimuat. "
        f"Device: {DEVICE}"
    )

    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


# ----------------------------------------------------------------------
# FUNGSI INFERENSI
# ----------------------------------------------------------------------

def predict_category(text: str) -> dict:
    """
    Melakukan inferensi terhadap satu teks artikel.

    Return dict berisi:
        - predicted_label: nama kategori dengan probabilitas tertinggi
        - predicted_id: id kelas (0/1/2)
        - confidence: probabilitas kelas terpilih (0-1)
        - probabilities: dict {nama_kelas: probabilitas} untuk semua kelas
    """
    if not text or not text.strip():
        raise ValueError("Teks input kosong.")

    # Tokenisasi, sama seperti saat training.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze(0)  # shape: (num_labels,)

    predicted_id = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_id].item())

    probabilities = {
        ID2LABEL[i]: round(float(probs[i].item()), 4) for i in range(len(ID2LABEL))
    }

    return {
        "predicted_label": ID2LABEL[predicted_id],
        "predicted_id": predicted_id,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }


def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """
    Mengekstrak seluruh teks dari file PDF menggunakan pypdf.

    Return tuple (teks_gabungan, jumlah_halaman).
    Raise ValueError kalau PDF tidak bisa dibaca atau tidak mengandung teks
    (misal PDF hasil scan tanpa OCR).
    """
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise ValueError(f"File PDF tidak bisa dibaca atau rusak: {str(e)}")

    page_count = len(reader.pages)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)

    full_text = "\n".join(text_parts).strip()

    if not full_text:
        raise ValueError(
            "Tidak ada teks yang bisa diekstrak dari PDF ini. "
            "Kemungkinan PDF berupa hasil scan/gambar tanpa lapisan teks (butuh OCR), "
            "yang belum didukung oleh sistem ini."
        )

    return full_text, page_count


# Penanda umum yang biasanya muncul tepat SETELAH bagian abstrak pada jurnal
# berbahasa Indonesia maupun Inggris. Dicari secara case-insensitive.
ABSTRACT_END_MARKERS = [
    "kata kunci", "keywords", "key words",
    "pendahuluan", "i. pendahuluan", "1. pendahuluan", "bab i",
    "introduction", "i. introduction", "1. introduction",
]

ABSTRACT_HARD_CHAR_LIMIT = 1500       # batas aman kalau penanda tidak ketemu
ABSTRACT_MARKER_SEARCH_WINDOW = 4000  # hanya cari penanda dalam N karakter pertama


def extract_abstract_preview(full_text: str) -> tuple[str, bool]:
    """
    Mengambil bagian abstrak saja dari teks hasil ekstraksi PDF, dengan
    pendekatan heuristik (bukan ekstraksi struktural yang sempurna):

    1. Cari penanda akhir abstrak (mis. "Kata Kunci", "Pendahuluan") di
       dalam beberapa ribu karakter pertama teks.
    2. Kalau ketemu, potong teks tepat sebelum penanda tersebut.
    3. Kalau tidak ketemu, fallback ke batas jumlah karakter tetap
       (dipotong di spasi terakhir supaya tidak memotong kata).

    Return tuple (teks_preview, is_truncated) — is_truncated menandakan
    apakah teks aslinya lebih panjang dari preview yang ditampilkan.
    """
    search_area = full_text[:ABSTRACT_MARKER_SEARCH_WINDOW]
    lower_area = search_area.lower()

    earliest_marker_idx = None
    for marker in ABSTRACT_END_MARKERS:
        idx = lower_area.find(marker)
        # idx > 30 untuk menghindari false-positive kalau kata itu kebetulan
        # muncul di judul/header paling awal.
        if idx > 30 and (earliest_marker_idx is None or idx < earliest_marker_idx):
            earliest_marker_idx = idx

    if earliest_marker_idx is not None and earliest_marker_idx <= ABSTRACT_HARD_CHAR_LIMIT:
        preview = full_text[:earliest_marker_idx].strip()
        is_truncated = len(full_text) > len(preview)
        return preview, is_truncated

    # Fallback: potong di batas karakter tetap, mundur ke spasi terdekat
    # supaya tidak memenggal kata di tengah.
    if len(full_text) <= ABSTRACT_HARD_CHAR_LIMIT:
        return full_text.strip(), False

    truncated = full_text[:ABSTRACT_HARD_CHAR_LIMIT]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.strip(), True


# ----------------------------------------------------------------------
# FLASK APP
# ----------------------------------------------------------------------

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "ganti-dengan-string-acak-untuk-produksi"  # dipakai untuk flash message

db.init_app(app)

with app.app_context():
    db.create_all()  # membuat tabel jika belum ada, tidak menimpa data yang sudah ada


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def api_predict():
    """
    Endpoint API untuk keperluan testing programatis / integrasi lanjutan.

    Contoh request (JSON):
        POST /predict
        {
            "text": "Isi artikel yang mau diklasifikasikan..."
        }

    Contoh response:
        {
            "predicted_label": "Teknologi",
            "predicted_id": 0,
            "confidence": 0.9123,
            "probabilities": {
                "Teknologi": 0.9123,
                "Medis": 0.0512,
                "Edukasi": 0.0365
            }
        }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' wajib dikirim dalam body JSON."}), 400

    try:
        result = predict_category(data["text"])
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # pragma: no cover
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def upload_page():
    """
    Halaman utama aplikasi: form untuk upload jurnal/artikel baru dalam format PDF.
    (Halaman 'Test Model' lama sudah dihapus; '/' sekarang langsung menuju upload.)
    """
    return render_template("upload.html")


@app.route("/", methods=["POST"])
def upload_submit():
    """
    Menangani submit file PDF baru:
    1. Validasi file (harus ada, harus .pdf).
    2. Simpan file PDF ke folder lokal `uploaded_pdfs/` dengan nama unik
       (supaya tidak bentrok kalau ada file dengan nama sama).
    3. Ekstrak teks dari PDF.
    4. Klasifikasikan teks hasil ekstraksi menggunakan model IndoBERT.
    5. Simpan record ke database (judul, path file, teks, hasil klasifikasi).
    """
    uploaded_file = request.files.get("file")
    title = request.form.get("title", "").strip()

    if not uploaded_file or not uploaded_file.filename:
        flash("Pilih file PDF terlebih dahulu.", "error")
        return redirect(url_for("upload_page"))

    if not allowed_file(uploaded_file.filename):
        flash("Format file tidak didukung. Hanya file .pdf yang diterima.", "error")
        return redirect(url_for("upload_page"))

    original_filename = secure_filename(uploaded_file.filename)

    # Nama file unik di disk supaya tidak menimpa file lain dengan nama sama.
    stored_filename = f"{uuid.uuid4().hex}_{original_filename}"
    stored_path = os.path.join(UPLOAD_FOLDER, stored_filename)
    uploaded_file.save(stored_path)

    # Ekstrak teks dari PDF yang baru disimpan.
    try:
        extracted_text, page_count = extract_text_from_pdf(stored_path)
    except ValueError as e:
        os.remove(stored_path)  # bersihkan file kalau ekstraksi gagal
        flash(str(e), "error")
        return redirect(url_for("upload_page"))

    # Kalau judul kosong, ambil dari nama file (tanpa ekstensi .pdf).
    if not title:
        title = os.path.splitext(original_filename)[0]

    # Klasifikasikan teks hasil ekstraksi.
    try:
        result = predict_category(extracted_text)
    except ValueError as e:
        os.remove(stored_path)
        flash(str(e), "error")
        return redirect(url_for("upload_page"))

    article = Article(
        title=title,
        content=extracted_text,
        original_filename=original_filename,
        stored_filename=stored_filename,
        page_count=page_count,
        category=result["predicted_label"],
        confidence=result["confidence"],
        prob_teknologi=result["probabilities"]["Teknologi"],
        prob_medis=result["probabilities"]["Medis"],
        prob_edukasi=result["probabilities"]["Edukasi"],
    )
    db.session.add(article)
    db.session.commit()

    flash(
        f"'{original_filename}' berhasil diupload dan diklasifikasikan sebagai "
        f"'{result['predicted_label']}' (confidence {result['confidence']*100:.1f}%).",
        "success",
    )
    return redirect(url_for("article_detail", article_id=article.id))


@app.route("/articles", methods=["GET"])
def article_list():
    """
    Menampilkan daftar artikel tersimpan.
    Mendukung filter kategori via query param ?category=Teknologi
    dan pencarian judul/isi via ?q=kata_kunci
    """
    category_filter = request.args.get("category", "").strip()
    search_query = request.args.get("q", "").strip()

    query = Article.query

    if category_filter and category_filter in ID2LABEL.values():
        query = query.filter(Article.category == category_filter)

    if search_query:
        like_pattern = f"%{search_query}%"
        query = query.filter(
            db.or_(
                Article.title.ilike(like_pattern),
                Article.content.ilike(like_pattern),
            )
        )

    articles = query.order_by(Article.uploaded_at.desc()).all()

    # Hitung jumlah artikel per kategori untuk ditampilkan di sidebar filter.
    category_counts = {
        label: Article.query.filter_by(category=label).count()
        for label in ID2LABEL.values()
    }

    return render_template(
        "articles.html",
        articles=articles,
        category_counts=category_counts,
        active_category=category_filter,
        search_query=search_query,
        total_count=Article.query.count(),
    )


@app.route("/articles/<int:article_id>", methods=["GET"])
def article_detail(article_id):
    """
    Menampilkan detail satu artikel: judul (bisa diedit), kategori,
    confidence, dan ABSTRAK saja (bukan isi PDF lengkap) di bagian
    "Isi Artikel". Naskah lengkap tetap bisa dibuka lewat file PDF asli.
    """
    article = Article.query.get_or_404(article_id)
    abstract_preview, is_truncated = extract_abstract_preview(article.content)
    return render_template(
        "article_detail.html",
        article=article,
        abstract_preview=abstract_preview,
        is_truncated=is_truncated,
    )


@app.route("/articles/<int:article_id>/edit-title", methods=["POST"])
def article_edit_title(article_id):
    """Mengubah judul artikel yang sudah tersimpan."""
    article = Article.query.get_or_404(article_id)
    new_title = request.form.get("title", "").strip()

    if not new_title:
        flash("Judul tidak boleh kosong.", "error")
        return redirect(url_for("article_detail", article_id=article_id))

    article.title = new_title
    db.session.commit()
    flash("Judul artikel berhasil diperbarui.", "success")
    return redirect(url_for("article_detail", article_id=article_id))


@app.route("/articles/<int:article_id>/pdf", methods=["GET"])
def article_pdf(article_id):
    """Menyajikan file PDF asli untuk dilihat/didownload di browser."""
    article = Article.query.get_or_404(article_id)
    return send_from_directory(
        UPLOAD_FOLDER,
        article.stored_filename,
        as_attachment=False,
        download_name=article.original_filename,
    )


@app.route("/articles/<int:article_id>/delete", methods=["POST"])
def article_delete(article_id):
    """Menghapus artikel dari database beserta file PDF fisiknya di disk."""
    article = Article.query.get_or_404(article_id)

    pdf_path = os.path.join(UPLOAD_FOLDER, article.stored_filename)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    db.session.delete(article)
    db.session.commit()
    flash("Artikel dan file PDF-nya berhasil dihapus.", "success")
    return redirect(url_for("article_list"))


if __name__ == "__main__":
    # debug=True hanya untuk keperluan development/testing skripsi,
    # matikan saat deploy.
    app.run(debug=True, host="127.0.0.1", port=5000)

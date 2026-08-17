---
title: Sistem Klasifikasi Jurnal IndoBERT
emoji: 📚
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Sistem Klasifikasi dan Pencarian Artikel Berbahasa Indonesia Menggunakan IndoBERT Berbasis Flask

Aplikasi web untuk mengklasifikasikan jurnal/artikel akademik berbahasa
Indonesia ke dalam 3 kategori (**Teknologi**, **Medis**, **Edukasi**)
menggunakan model IndoBERT yang di-fine-tune, dengan fitur upload PDF,
ekstraksi teks otomatis, dan penyimpanan lokal.

**Bagian YAML di atas** adalah konfigurasi khusus Hugging Face Spaces
(SDK: Docker, port: 7860) — JANGAN dihapus atau diedit strukturnya, karena
dipakai oleh Hugging Face untuk mem-build & menjalankan Space ini.

Untuk dokumentasi lengkap proyek ini, lihat `PANDUAN.md`.
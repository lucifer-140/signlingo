# Laporan Project: SignLingo - Deteksi Bahasa Isyarat

## Deskripsi Singkat Project

**SignLingo** adalah sebuah aplikasi berbasis Python yang dirancang untuk mendeteksi dan menerjemahkan bahasa isyarat secara real-time menggunakan input video dari webcam. Aplikasi ini bertujuan untuk menjembatani komunikasi bagi penyandang tunarungu atau siapa saja yang ingin mempelajari bahasa isyarat dengan memberikan umpan balik visual langsung.

### Cara Kerja Aplikasi
Aplikasi ini bekerja dengan langkah-langkah berikut:
1.  **Input Video**: Mengambil feed video langsung dari webcam pengguna.
2.  **Ekstraksi Fitur**: Menggunakan library **MediaPipe Holistic** untuk mendeteksi titik-titik kunci (keypoints) pada wajah, tangan, dan pose tubuh pengguna.
3.  **Preprocessing**: Data titik kunci dinormalisasi dan disusun menjadi urutan (sequence) data temporal.
4.  **Klasifikasi**: Model **LSTM (Long Short-Term Memory)** yang telah dilatih memproses urutan data tersebut untuk memprediksi kata isyarat yang sedang diperagakan.
5.  **Output**: Menampilkan kata hasil prediksi dan tingkat kepercayaan (confidence score) di layar jika melewati ambang batas tertentu.

### Rencana Pengembangan Selanjutnya
Untuk pengembangan di masa depan, kami berencana untuk:
*   Menambah jumlah kosakata bahasa isyarat yang dapat dideteksi.
*   Mengembangkan antarmuka pengguna (GUI) yang lebih ramah pengguna.
*   Mengintegrasikan fitur *text-to-speech* agar hasil terjemahan dapat didengar.
*   Membuat versi mobile dari aplikasi ini.

## Hasil Tampilan Project

Berikut adalah tampilan antarmuka aplikasi saat dijalankan, menunjukkan deteksi kerangka tangan dan tubuh serta hasil prediksi kata:

![Tampilan Aplikasi SignLingo](screenshot_mockup.png)

## Cara Menjalankan Aplikasi

Untuk menjalankan aplikasi ini di komputer lokal, ikuti langkah-langkah berikut:

1.  **Persiapan Lingkungan**:
    Pastikan Python 3.8+ sudah terinstal. Disarankan menggunakan virtual environment.
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

2.  **Instalasi Dependensi**:
    Instal library yang dibutuhkan (seperti opencv-python, mediapipe, torch, numpy).
    ```bash
    pip install opencv-python mediapipe torch numpy
    ```

3.  **Menjalankan Aplikasi**:
    Jalankan script inferensi utama melalui terminal:
    ```bash
    python scripts/infer_webcam.py
    ```
    *   Tekan `ESC` untuk keluar.
    *   Tekan `t` untuk menyembunyikan/menampilkan tracking lines.
    *   Tekan `b` untuk menyembunyikan/menampilkan bounding box.

## Kesimpulan dan Saran

### Kesimpulan
Selama pengerjaan project UAS ini, kami berhasil mengembangkan prototipe fungsional sistem deteksi bahasa isyarat. Penggunaan model LSTM terbukti cukup efektif untuk mengenali urutan gerakan tangan yang dinamis. Tantangan utama yang dihadapi adalah variasi pencahayaan dan kecepatan gerakan tangan yang dapat mempengaruhi akurasi deteksi MediaPipe.

### Saran
Saran untuk pengembangan selanjutnya atau bagi yang ingin membuat project serupa:
*   Perbanyak variasi data latih dengan berbagai kondisi pencahayaan dan latar belakang untuk meningkatkan ketahanan model.
*   Eksplorasi arsitektur model lain seperti Transformer atau GRU untuk membandingkan performa dan efisiensi.

## Resource atau Dokumentasi

Hasil pengerjaan project ini dapat diakses melalui link publik berikut:
*   **Repository & Dokumentasi**: [LINK_REPOSITORY_ANDA_DISINI]
*   **Video Demo**: [LINK_VIDEO_DEMO_ANDA_DISINI]

## Kontribusi dan Harapan

### Kontribusi Anggota
Usaha yang kami berikan dalam project ini sangat maksimal, dengan pembagian tugas sebagai berikut:

*   **Dave (Saya)**:
    *   Bertanggung jawab atas implementasi model *Deep Learning* (LSTM) dan *training pipeline*.
    *   Menulis script untuk preprocessing data dan ekstraksi fitur menggunakan MediaPipe.
    *   Menyusun laporan teknis dan dokumentasi kode.
    *   *Estimasi Usaha: 50%*

*   **Alsando**:
    *   Bertanggung jawab atas pengumpulan data video (rekaman gerakan isyarat).
    *   Mengembangkan script antarmuka webcam (`infer_webcam.py`) dan visualisasi (HUD).
    *   Melakukan pengujian aplikasi dan *debugging* sistem secara keseluruhan.
    *   *Estimasi Usaha: 50%*

### Harapan Nilai
Mengingat kompleksitas teknologi yang digunakan (Computer Vision & Deep Learning), usaha yang kami curahkan untuk riset dan implementasi, serta hasil aplikasi yang berfungsi dengan baik secara real-time, kami berharap dapat memperoleh nilai **A** pada matakuliah ini. Kami percaya project ini menunjukkan pemahaman yang mendalam terhadap materi yang diajarkan.

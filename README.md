# 🌌 Space Gesture Controller - Real-time Hand Tracking & Particle System

Aplikasi Python interaktif untuk mengontrol simulasi sistem partikel dinamis luar angkasa secara *real-time* menggunakan gestur tangan melalui webcam.

---

## 🌟 Fitur Utama & Arsitektur

### 1. Sistem Multi-Window Terintegrasi
- **Jendela 1: "Hand Sensor Monitor" (OpenCV)**
  - Umpan langsung kamera webcam dalam mode cermin (*mirrored feed*).
  - Visualisasi skeleton & landmark 21 titik jari tangan dari **MediaPipe Hands**.
  - Modern HUD Glassmorphism di pojok kiri atas yang menampilkan:
    * Status gestur yang terdeteksi secara real-time.
    * Indikator FPS dan status sensor deteksi tangan.
- **Jendela 2: "Space Gesture Controller" (Pygame)**
  - Canvas luar angkasa hitam pekat dengan **2.500 partikel dinamis**.
  - Animasi transisi fisik halus (**Damped Spring & Smooth Lerp Physics**) saat bertransformasi antar bentuk.
  - Efek pencahayaan warna dinamis (*color interpolation*) dan jejak gerak (*motion trail*).
  - Interaksi posisi tangan: Partikel merespons pergerakan posisi tangan (*interactive parallax*).

---

## 🖐️ Logika Deteksi Gestur & Formasi Partikel

| No | Gestur Tangan | Tampilan HUD | Formasi Partikel | Warna | Efek Animasi |
|---|---|---|---|---|---|
| 1 | **Tangan Terbuka** (*Open Palm*) | `MODE: KOSMOS & BINTANG (Tangan Terbuka)` | Nebula Galaksi Spiral / Taburan Bintang | Biru Elektrik & Cyan (`#2DAFFF`) | Gerakan rotasi melayang lembut (*swirling celestial drift*) |
| 2 | **Tangan Mengepal** (*Fist*) | `MODE: JANTUNG HATI (Mengepal)` | Siluet Jantung Hati Parametrik 2D/3D | Pink Romantis & Magenta (`#FF2D87`) | Efek denyut ritmis jantung (*heartbeat pulse*) |
| 3 | **Dua Jari / Peace** (*Index + Middle*) | `MODE: TEKS I LOVE YOU (Dua Jari)` | Tipografi Teks "I LOVE YOU" | Cyan Neon & Turquoise (`#00EBFF`) | Efek gelombang mengambang halus (*floating wave shimmer*) |
| 4 | **Satu Jari Menunjuk** (*Pointing Index*) | `MODE: PLANET SATURNUS (Satu Jari)` | Planet Bola 3D + Piringan Cincin | Emas Hangat & Jingga Kosmik (`#FF9C19`) | Rotasi 3D kontinu dengan kemiringan sumbu aksial (*axial tilt projection*) |
| 5 | **Gestur Cinta 🤟 / 3 Jari** (*ASL Love Sign / 3 Fingers*) | `MODE: I LOVE YOU ALYA (Gestur Cinta 🤟)` | Tipografi Teks **"I LOVE YOU ALYA"** + Dual Sparkling Hearts | Neon Cyan & Valentine Rose Pink (`#FF37A0`) | Denyut cinta pada nama ALYA, gelombang teks, dan kilau hati (*heart pulse & sparkle*) |

---

## 📁 Struktur Direktori

```
KameraTracking/
├── main.py              # Entry point aplikasi & sinkronisasi multi-window loop
├── hand_tracker.py      # Modul deteksi MediaPipe, klasifikasi gestur, & HUD
├── particle_system.py   # Modul mesin fisika partikel NumPy, 5 formasi, & rotasi 3D
├── requirements.txt     # Daftar dependensi pustaka Python
├── verify_system.py     # Skrip pengujian otomatis unit & performa
└── run.bat              # File batch peluncur cepat di Windows
```

---

## 🚀 Panduan Instalasi & Menjalankan

### 1. Prasyarat
Pastikan Python 3.10, 3.11, atau 3.12 sudah terpasang di komputer Anda.

### 2. Instalasi Dependensi
Buka terminal / PowerShell di direktori proyek ini, lalu jalankan:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
Jalankan perintah:
```bash
python main.py
```
*Atau klik ganda file `run.bat` di Windows Explorer.*

---

## ⌨️ Kontrol Pintasan Keyboard
Selain menggunakan webcam, Anda juga dapat menguji atau mengontrol formasi secara manual menggunakan keyboard:
- `1` : Beralih ke formasi **Kosmos / Bintang**
- `2` : Beralih ke formasi **Jantung Hati**
- `3` : Beralih ke formasi teks **I LOVE YOU**
- `4` : Beralih ke formasi **Planet Saturnus 3D**
- `5` : Beralih ke formasi spesial **I LOVE YOU ALYA** (🤟)
- `Q` atau `ESC` : Keluar dari aplikasi dengan aman

# 🛰️ Simulasi Orbit Satelit Multi-Benda

Simulasi orbit satelit 2D berbasis Python dengan autopilot Hohmann, peluncuran dari permukaan, gravitasi tiga benda (Bumi–Satelit–Bulan/Mars), dan antarmuka grafis interaktif.

Dikembangkan sebagai proyek riset komputasional oleh **Kelompok 1, Jurusan Fisika FMIPA Universitas Sam Ratulangi Manado**.

---

## 📸 Tampilan

> *(Screenshot antarmuka utama dan manuver orbit tersedia di folder `/docs/`)*

---

## ✨ Fitur Utama

- **Integrator DOP853** (Dormand-Prince orde-8) via `scipy.solve_ivp` dengan fallback RK4 sub-stepping adaptif
- **Gravitasi tiga benda**: Bumi–Satelit–Bulan, dengan posisi Bulan diambil dari efemerida `astropy` secara real-time (cache 3600 detik)
- **Autopilot Transfer Hohmann** ke Bulan atau Mars dengan deteksi fase optimal iteratif (|ε| < 0.01 rad)
- **Autopilot Peluncuran** dari permukaan Bumi (altitude 0 km) ke orbit parkir 325 km dengan profil thrust adaptif (fase vertikal + fase ascent)
- **Manuver Kembali ke Bumi** menuju orbit aman 4000 km (burn retrograde bertahap + coast + sirkularisasi)
- **Mode Rendezvous / Flyby** yang dapat diubah saat runtime
- **Grafik Energi Live**: jendela matplotlib terpisah menampilkan KE, PE, dan TE secara real-time (update 250 ms) dengan penanda momen manuver
- **>40 pintasan keyboard** untuk kontrol penuh tanpa mouse
- **Ghost orbit preview** sebelum eksekusi manuver
- Penghitung waktu simulasi dan total Δv kumulatif

---

## 🔧 Persyaratan

Python 3.8 atau lebih baru.

```
numpy
scipy >= 1.7
astropy >= 5.0
matplotlib >= 3.5
tkinter  # sudah termasuk di instalasi Python standar
```

Install dependensi:

```bash
pip install numpy scipy astropy matplotlib
```

---

## 🚀 Cara Menjalankan

```bash
python satelite.py
```

Tidak diperlukan argumen tambahan. Simulasi akan langsung berjalan dengan kondisi awal alt 0 km.

---

## ⌨️ Panduan Shortcut Keyboard

### 1. Kontrol Simulasi Utama
| Tombol | Fungsi |
|--------|--------|
| `Space` | Toggle Jalankan / Pause |
| `R` | Reset simulasi |
| `M` | Tembak Manuver |
| `G` | Toggle Ghost orbit On/Off |
| `O` | Toggle Garis Orbit |
| `L` | Toggle Lock Zoom |

### 2. Thruster Manual
| Tombol | Fungsi |
|--------|--------|
| `Z` (tahan) | BURN aktif |
| `,` (koma) | Arah burn −5° |
| `.` (titik) | Arah burn +5° |
| `[` | Kurangi Δv rate (−10 m/s) |
| `]` | Tambah Δv rate (+10 m/s) |
| `1` | Preset Prograde (0°) |
| `2` | Preset Normal (90°) |
| `3` | Preset Retrograde (180°) |
| `4` | Preset Radial-in (270°) |

### 3. Zoom & Pan
| Tombol | Fungsi |
|--------|--------|
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `↑ ↓ ← →` | Pan canvas |
| `BackSpace` | Reset zoom |

### 4. Slider Parameter
| Tombol | Fungsi |
|--------|--------|
| `Q` / `W` | Altitude awal −/+ |
| `A` / `S` | Eksentrisitas awal −/+ |
| `Y` / `U` | Altitude manuver −/+ |
| `H` / `J` | Eksentrisitas manuver −/+ |
| `D` (tahan) | dt simulasi −1 |
| `F` (tahan) | dt simulasi +1 |
| `Ctrl+,` | Kecepatan sim −1 |
| `Ctrl+.` | Kecepatan sim +1 |

### 5. Autopilot & Navigasi
| Tombol | Fungsi |
|--------|--------|
| `N` | Autopilot ke Bulan |
| `P` | Autopilot ke Mars |
| `X` | Batal Autopilot |
| `B` | Toggle Mode Rendezvous / Flyby |
| `K` | Kembali ke Bumi |

### 6. Launch
| Tombol | Fungsi |
|--------|--------|
| `V` | Launch ke orbit 325 km (hanya aktif saat di permukaan & sim berhenti) |
| `C` | Batal Launch |

### 7. Zoom Cepat, Grafik & Riwayat
| Tombol | Fungsi |
|--------|--------|
| `5` | Toggle Fokus ke Satelit |
| `6` | Zoom ke Bulan |
| `7` | Zoom ke Mars |
| `8` | Buka Grafik Energi live (KE/PE/TE) |
| `9` | Hapus Riwayat orbit |
| `0` | Toggle Gravitasi Bulan ON/OFF |

---

## ⚙️ Parameter Fisika

| Parameter | Simbol | Nilai |
|-----------|--------|-------|
| Konstanta gravitasi | G | 6.67 × 10⁻¹¹ m³·kg⁻¹·s⁻² |
| Massa Bumi | M | 5.97 × 10²⁴ kg |
| Jari-jari Bumi | R | 6.371 × 10⁶ m |
| Kecepatan sudut Bumi | Ω | 7.2921 × 10⁻⁵ rad/s |
| Massa Bulan | M☽ | 7.342 × 10²² kg |
| Jari-jari Bulan | R☽ | 1.7374 × 10⁶ m |
| Massa satelit (referensi) | m | 1000 kg |
| Altitude target return | — | 4000 km |

---

## 📐 Arsitektur Kode

```
satelite.py
├── Konstanta fisika & utilitas
│   ├── v_circular(r)
│   ├── rk4_step()             — Integrator RK4 klasik (fallback)
│   ├── rk4_step_safe()        — DOP853 via solve_ivp + fallback RK4
│   ├── orbital_elements()     — Elemen Keplerian dari state vektor
│   ├── periapsis_apoapsis()
│   ├── predict_orbit_full()   — Ghost orbit preview
│   └── draw_*()               — Utilitas rendering Tkinter
│
└── class OrbitApp             — Kelas utama aplikasi
    ├── Subsistem propagator fisika (DOP853 + RK4)
    ├── Subsistem efemerida astropy (Bulan & Mars)
    ├── Autopilot Hohmann (waiting → burn1 → coast → burn2)
    ├── Autopilot Launch (vertical → ascent adaptif)
    ├── Manuver Return to Earth (wait_apoapsis → burn3 → coast → burn4)
    ├── Antarmuka Tkinter (kanvas orbit, panel info, slider)
    └── Grafik Energi live (matplotlib embedded)
```

---

## 📊 Validasi Numerik

Konservasi energi mekanik total pada orbit Keplerian (tanpa gravitasi Bulan):

| Kondisi Orbit | Eksentrisitas | ΔE/E₀ per periode |
|---------------|--------------|-------------------|
| Sirkular 325 km | 0.000 | 8.2 × 10⁻⁶ |
| Elips sedang | 0.200 | 9.5 × 10⁻⁶ |
| Elips tinggi | 0.600 | 1.1 × 10⁻⁵ |

Semua di bawah ambang validasi **< 0.01%** per periode.

---

## 📄 Publikasi

Simulasi ini didokumentasikan dalam artikel penelitian:

> Faizan, N., Tasiam, F.A., Syafia, M.R.A.P., Pantouw, N., & Pandara, D.P. (2026).
> *Simulasi Orbit Satelit Multi-Benda dengan Autopilot Hohmann, Launch dari Permukaan, Manuver Kembali ke Bumi, dan Gravitasi Tiga Benda Menggunakan Integrator DOP853 dan Astropy.*
> Jurnal Fisika dan Terapannya, FMIPA UNSRAT.

---

## 👥 Tim Pengembang

**Kelompok 1 — Jurusan Fisika, FMIPA, Universitas Sam Ratulangi Manado**

- Nasrul Faizan
- Fiorenza Abygail Tasiam
- Mohamad Rayan Akbar Putra Syafia
- Nadine Pantouw
- Dolfie Paulus Pandara (Pembimbing)

Korespondensi: `nasrulfaizan104@student.unsrat.ac.id`

---

## 📚 Referensi

- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I*. Springer.
- Virtanen, P., et al. (2020). SciPy 1.0. *Nature Methods*, 17, 261–272.
- Astropy Collaboration (2022). *The Astrophysical Journal*, 935(2), 167.
- Curtis, H. D. (2014). *Orbital Mechanics for Engineering Students* (3rd ed.). Elsevier.

---

## 📝 Lisensi

© 2026 Kelompok 1, Jurusan Fisika FMIPA UNSRAT. Semua hak dilindungi.

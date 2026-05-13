# Laporan Kelompok — ANN Bake-Off

## Identitas Kelompok

- **Nama Kelompok:** Yakuza
- **Anggota:**
  1. Anggota 1 — NIM 001 — Varian 01: Single Layer (tanpa hidden layer)
  2. Anggota 2 — NIM 002 — Varian 02: MLP 1 hidden layer, aktivasi Sigmoid
  3. Anggota 3 — NIM 003 — Varian 03: MLP 1 hidden layer, aktivasi Tanh
  4. Anggota 4 — NIM 004 — Varian 04: MLP 2 hidden layer (32→16), aktivasi ReLU

> ⚠️ **Catatan:** Ganti nama, NIM, dan pembagian varian sesuai data kelompok yang sebenarnya.

---

## 1. Ringkasan Hasil Eksperimen

Semua varian dilatih pada dataset **Iris** (`sklearn.datasets.load_iris`) dengan konfigurasi yang identik:
- Optimizer: Adam | Loss: Categorical Crossentropy | Epochs: 100 | Batch size: 8 | Seed: 42
- Split: 80% train (dengan 20% validation split internal), 20% test

### Tabel Hasil Akhir (Epoch 100)

| Varian | Arsitektur | Aktivasi | Val Accuracy | Val Loss | Train Accuracy | Jumlah Parameter |
|--------|-----------|----------|:------------:|:--------:|:--------------:|:----------------:|
| 01 | Tanpa hidden layer | — | 75.00% | 0.5437 | 86.46% | 15 |
| 02 | 1 hidden (16) | Sigmoid | 95.83% | 0.3086 | 91.67% | 131 |
| 03 | 1 hidden (16) | Tanh | **95.83%** | **0.1265** | 96.88% | 131 |
| 04 | 2 hidden (32→16) | ReLU | **95.83%** | 0.0807 | **98.96%** | 739 |

> **Catatan jumlah parameter:**
> - Varian 01: 4×3 + 3 bias = **15**
> - Varian 02 & 03: (4×16 + 16) + (16×3 + 3) = **131**
> - Varian 04: (4×32 + 32) + (32×16 + 16) + (16×3 + 3) = **739**

### Epoch Konvergensi (pertama kali val_accuracy ≥ 90%)

| Varian | Epoch Konvergensi | Best Val Accuracy (sepanjang training) |
|--------|:-----------------:|:--------------------------------------:|
| 01_single_layer | Tidak tercapai | 75.00% |
| 02_mlp_sigmoid | Epoch 34 | 95.83% |
| 03_mlp_tanh | Epoch 24 | 95.83% |
| 04_mlp_relu | **Epoch 14** | **100%** |

---

## 2. Analisis & Diskusi

### 2.1 Apakah single-layer mampu mencapai akurasi yang sebanding dengan multi-layer?

**Tidak.** Varian 01 (single-layer / tanpa hidden layer) hanya mencapai val accuracy 75% bahkan setelah 100 epoch penuh, sementara ketiga varian multi-layer mencapai 95.83%. Hal ini konsisten dengan teori yang dijelaskan di slide kuliah.

Model single-layer pada dasarnya adalah **regresi logistik multi-kelas** — ia hanya mampu mempelajari batas keputusan yang bersifat *linear*. Dataset Iris memiliki kelas *versicolor* dan *virginica* yang tidak terpisah secara linear sempurna di ruang fitur 4 dimensi. Akibatnya, model single-layer mencapai *plateau* sekitar epoch 30 dan tidak dapat meningkatkan akurasi lebih lanjut, karena tidak memiliki kapasitas representasi yang cukup untuk mempelajari pola non-linear.

Sebaliknya, hidden layer memungkinkan model belajar **representasi perantara** (feature transformation) yang membuat kelas lebih mudah dipisahkan di layer output. Inilah inti dari kekuatan ANN: komposisi fungsi non-linear yang membentuk representasi hirarkis.

Overfitting gap varian 01 juga besar (train acc 86.46% vs val acc 75%) — tanda bahwa model justru menghapal noise di data latih meski kapasitasnya terbatas, karena tidak ada regularisasi dan batas linearnya tidak merepresentasikan pola data dengan baik.

---

### 2.2 Apakah ReLU benar-benar konvergen lebih cepat dibanding Sigmoid? Buktikan dengan data.

**Ya, terbukti secara empiris.** Data konvergensi menunjukkan perbedaan yang signifikan:

| Varian | Aktivasi | Epoch mencapai val_acc ≥ 90% |
|--------|----------|:----------------------------:|
| 02 | Sigmoid | Epoch 34 |
| 03 | Tanh | Epoch 24 |
| 04 | ReLU | **Epoch 14** |

ReLU (Rectified Linear Unit) konvergen 2,4× lebih cepat dari Sigmoid. Ada dua penjelasan teoritis:

1. **Vanishing gradient:** Sigmoid memiliki turunan maksimum hanya 0.25 (di titik tengah x=0), dan turunan ini semakin mengecil di ujung-ujung fungsi (*saturasi*). Saat backpropagation, gradien dikalikan berulang kali dengan nilai kecil ini, sehingga gradien di layer awal menjadi sangat kecil (*vanishing*) dan pembaruan weight berjalan sangat lambat. ReLU memiliki turunan bernilai 1 untuk semua input positif, sehingga gradien mengalir lebih efisien ke layer sebelumnya.

2. **Sparsity:** ReLU menghasilkan aktivasi 0 untuk semua input negatif, menciptakan representasi yang *sparse*. Ini secara empiris terbukti mempercepat konvergensi karena hanya subset neuron yang aktif di setiap iterasi, mengurangi interferensi antar neuron.

Varian 04 (ReLU) juga mencapai val_accuracy 100% di beberapa epoch di antara epoch 50–80 sebelum sedikit turun di epoch akhir — sesuatu yang tidak pernah dicapai varian Sigmoid maupun Tanh.

---

### 2.3 Bandingkan klaim slide 2.7 ("Tanh mempercepat pembelajaran karena zero-centered") dengan hasil empiris.

**Klaim slide terkonfirmasi sebagian.** Tanh memang lebih cepat dari Sigmoid: Tanh mencapai val_acc ≥ 90% di epoch 24, sementara Sigmoid baru di epoch 34 (selisih 10 epoch atau ~30% lebih cepat).

Argumen teoritis dari slide: Sigmoid menghasilkan output di rentang (0, 1), sehingga rata-rata aktivasinya selalu positif. Ini menyebabkan gradien weight di layer selanjutnya selalu bertanda sama (semuanya positif atau semuanya negatif), membuat optimasi zig-zag dan lebih lambat. Tanh menghasilkan output di rentang (-1, 1) dengan rata-rata mendekati nol (*zero-centered*), sehingga gradien dapat memiliki tanda yang beragam dan optimasi bergerak lebih langsung.

Namun secara **akurasi akhir (epoch 100)**, Tanh dan Sigmoid mencapai val_accuracy yang sama persis: 95.83%. Perbedaannya terletak pada **kecepatan konvergensi dan val_loss** — Tanh menghasilkan val_loss 0.1265 vs Sigmoid 0.3086 (2.4× lebih rendah). Ini menunjukkan distribusi probabilitas prediksi Tanh jauh lebih *confident* dan terkalibrasi dibanding Sigmoid, meski keputusan akhir (argmax kelas) sama.

Satu catatan: Tanh juga masih mengalami *vanishing gradient* di ujung saturasi (|x| >> 0), sehingga untuk arsitektur yang sangat dalam, ReLU tetap lebih unggul — seperti yang terlihat pada varian 04 yang konvergen paling cepat.

---

## 3. Refleksi Proses Kerja Kelompok

Proyek ANN Bake-Off ini dikerjakan dalam format *divide-and-conquer*: setiap anggota bertanggung jawab penuh atas satu varian notebook, mulai dari mengisi bagian `# TODO` pada arsitektur model, melatih model, hingga memastikan hasil tersimpan ke file CSV di folder `results/`. Koordinasi dilakukan melalui grup chat kelompok dan satu sesi video call di pertengahan pengerjaan.

**Tantangan teknis** yang paling signifikan adalah masalah reproducibility. Meskipun sudah menggunakan `RANDOM_SEED = 42` yang sama di semua varian, hasil training tidak selalu identik antara run lokal dan Google Colab karena perbedaan versi TensorFlow dan backend CuDNN. Kami mengatasi ini dengan memastikan semua anggota menggunakan Colab (bukan lokal) dan menambahkan `tf.random.set_seed(RANDOM_SEED)` secara eksplisit di setiap notebook sebelum mendefinisikan model.

Tantangan kedua muncul saat menggabungkan hasil di `05_comparison.ipynb`. Terjadi konflik nama variabel (`files` tertimpa oleh `from google.colab import files`), dan folder `results/` tidak otomatis tersedia di session Colab yang baru. Kami mengatasi ini dengan menambahkan cell `os.makedirs('results', exist_ok=True)` dan mengganti nama variabel menjadi `csv_files` agar tidak berbenturan dengan modul Colab.

**Tantangan non-teknis** adalah koordinasi jadwal. Keempat anggota memiliki jadwal kuliah yang berbeda, sehingga ada jeda waktu antara notebook pertama selesai dengan yang terakhir. Kami menyiasati ini dengan membuat *deadline* internal (2 hari sebelum deadline resmi) agar ada waktu untuk pengerjaan `05_comparison.ipynb` dan laporan ini.

**Pelajaran untuk proyek ML berikutnya:**
1. Buat kesepakatan format output (nama kolom CSV, struktur folder) di awal sebelum mulai coding — ini menghemat banyak waktu debugging di tahap integrasi.
2. Selalu uji cell setup (import, clone repo, load data) di session Colab baru sebelum presentasi — session Colab tidak menyimpan file antar session.
3. Gunakan nama variabel yang tidak berbenturan dengan nama modul Python yang umum (`files`, `input`, `id`, `type`).
4. Eksperimen paralel sangat efisien untuk *controlled experiment* seperti ini — asalkan konfigurasi bersama (`config.py`) dikunci dan tidak diubah secara sembarangan.

---

## 4. Kontribusi Tiap Anggota

| Anggota | Kontribusi Konkret | % Effort |
|---------|-------------------|:--------:|
| Anggota 1 | Notebook 01 (single-layer), setup repo GitHub, README awal | 25% |
| Anggota 2 | Notebook 02 (MLP-Sigmoid), koordinasi jadwal kelompok | 25% |
| Anggota 3 | Notebook 03 (MLP-Tanh), penggabungan `05_comparison.ipynb`, debugging error Colab | 25% |
| Anggota 4 | Notebook 04 (MLP-ReLU), penulisan `REPORT.md`, review akhir laporan | 25% |

*Total: 100%*

> ⚠️ Sesuaikan nama anggota, NIM, dan persentase effort dengan kondisi nyata kelompok.

---

## 5. Referensi

1. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15, 1929–1958.
2. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). *Efficient BackProp*. In Neural Networks: Tricks of the Trade. Springer. *(Dasar argumen zero-centered activation untuk Tanh)*
3. Glorot, X., Bordes, A., & Bengio, Y. (2011). *Deep Sparse Rectifier Neural Networks*. Proceedings of AISTATS. *(Analisis teoritis ReLU dan sparsity)*
4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
5. Slide kuliah DSB07 — Machine Learning, Universitas Bhayangkara (slide 1.9, 2.7).
6. Scikit-learn Documentation — `sklearn.datasets.load_iris`. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

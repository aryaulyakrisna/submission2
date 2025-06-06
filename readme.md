# Laporan Proyek Machine Learning - Arya Ulya Krisna

### [github - aryaulyakrisna](https://github.com/aryaulyakrisna)

## Project Overview

Sistem rekomendasi buku merupakan solusi penting untuk membantu pengguna menemukan buku yang sesuai dengan minat mereka. Proyek ini bertujuan untuk mengembangkan sistem rekomendasi buku berbasis *content-based filtering* menggunakan algoritma *TF-IDF* dan *Cosine Similarity*. Sistem ini memanfaatkan metadata buku, seperti judul dan penulis, untuk memberikan rekomendasi berdasarkan kesamaan konten. Permasalahan ini relevan karena pembaca sering kesulitan menemukan buku yang sesuai akibat kurangnya informasi, sebagaimana dijelaskan dalam penelitian Ardiansyah et al. (2023). Penelitian tersebut menunjukkan bahwa sistem rekomendasi berbasis *content-based filtering* efektif dalam mengelola informasi buku dan memberikan rekomendasi yang relevan berdasarkan preferensi pengguna. Proyek ini menggunakan dataset dari Kaggle untuk membangun sistem yang dapat diimplementasikan pada platform digital guna meningkatkan aksesibilitas dan pengalaman pengguna.

**Referensi**:
- Ardiansyah, R., Saputra, B. D., & Bianto, M. A. (2023). Sistem rekomendasi buku perpustakaan sekolah menggunakan metode content-based filtering. *Jurnal Computer Science and Information Technology (CoSciTech)*, 4(2), 510-517. https://doi.org/10.37859/coscitech.v4i2.5131

## Business Understanding

### Problem Statements
1. Pembaca mengalami kesulitan menemukan buku yang sesuai dengan minat mereka karena keterbatasan informasi tentang koleksi buku.
2. Perlu adanya metode evaluasi untuk memastikan rekomendasi buku yang diberikan relevan dan akurat.

### Goals
1. Mengembangkan sistem rekomendasi buku berbasis *content-based filtering* untuk memberikan rekomendasi berdasarkan metadata buku seperti judul dan penulis.
2. Mengevaluasi performa sistem rekomendasi menggunakan metrik *precision*, *recall*, dan *F1-score* untuk memastikan kualitas rekomendasi.

### Solution Approach
1. Menerapkan algoritma *TF-IDF* untuk mengubah metadata buku menjadi representasi numerik, diikuti dengan *Cosine Similarity* untuk menghitung kesamaan antar buku, sebagaimana digunakan dalam penelitian Ardiansyah et al. (2023).
2. Mengevaluasi sistem menggunakan metrik *precision*, *recall*, dan *F1-score* untuk mengukur relevansi rekomendasi.

## Data Understanding

Proyek ini menggunakan *Book Recommendation Dataset* dari Kaggle, yang tersedia di [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Dataset ini terdiri dari dua file utama:
- **Books.csv**: Berisi metadata buku dengan sekitar 271,360 entri.
- **Ratings.csv**: Berisi data penilaian pengguna terhadap buku dengan sekitar 1,149,780 entri.

### Variabel-variabel pada Dataset
1. **Books.csv**:
   - **ISBN**: Kode unik untuk setiap buku (string).
   - **Book-Title**: Judul buku (string).
   - **Book-Author**: Nama penulis buku (string).
   - **Year-Of-Publication**: Tahun publikasi buku (campuran string dan integer, perlu pembersihan).
   - **Publisher**: Nama penerbit buku (string).
   - **Image-URL-S/M/L**: URL gambar sampul buku dalam tiga ukuran (string).

2. **Ratings.csv**:
   - **User-ID**: ID unik pengguna yang memberikan rating (integer).
   - **ISBN**: Kode unik buku yang diberi rating (string).
   - **Book-Rating**: Nilai rating buku (skala 0-10, integer).
  
### Kondisi Dataset

![Screenshot 2025-06-04 160346](https://github.com/user-attachments/assets/4da4f9a3-d7cb-4217-8393-b36edc2437dd)
- Kedua dataset tidak memiliki duplicated row

![Screenshot 2025-06-07 143552](https://github.com/user-attachments/assets/e340eb42-9ec8-4877-adc6-5d73bdc54657)
- Dataframe books berisi kolom bertipe object kecuali Year-Of-Publication, tampak beberapa nilai null pada kolom Book-Author dan Publisher.

![Screenshot 2025-06-07 143943](https://github.com/user-attachments/assets/64246a17-b5c8-4a50-a6ca-a1e17e200638)
- Dataframe ratings berisi kolom bertipe int64 dan object, tanpa nilai null.

![Screenshot 2025-06-04 155501](https://github.com/user-attachments/assets/e8820e16-b708-42ca-8eef-378541be565d)
- Setelah dataset digabungkan berdasarkan ISBN, dataset memiliki 118648 kolom null pada kolom Book-Title Book-Authorn Year-Of-Publication Publisher.

### Exploratory Data Analysis
- Kolom *Year-Of-Publication* pada *Books.csv* memiliki masalah tipe data, dengan beberapa entri berupa string (misalnya, nama penerbit yang salah tempat), menyebabkan *DtypeWarning* saat memuat data.
- Data *Ratings.csv* menunjukkan distribusi rating yang tidak merata, dengan banyak buku memiliki rating 0, yang mungkin menunjukkan rating implisit atau tidak adanya penilaian.

## Data Preparation

Tahapan persiapan data yang dilakukan adalah sebagai berikut:

1. **Pemrosesan & Pemrosesan Data**:
   ``` python
     books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
     books = books.dropna(subset=['Year-Of-Publication'])
     books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
   ```
   
   - Mengatasi *DtypeWarning* pada *Year-Of-Publication* dengan mengonversi kolom ke tipe numerik menggunakan `pd.to_numeric` dengan parameter `errors='coerce'` untuk mengubah nilai non-numerik menjadi *NaN*.
   - Menghapus baris dengan nilai *NaN* pada *Year-Of-Publication* menggunakan `dropna`.
   - Mengonversi *Year-Of-Publication* ke tipe integer untuk konsistensi.
  
   ``` python
      books.drop(labels=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
   ```
   - Menghapus kolom Image-URL-S, Image-URL-M, Image-URL-L
   
  
   ```python
      books_df = pd.merge(ratings, books, on='ISBN', how='left')
   ```
   - Menggabungkan kolom *Book-Title* dan *Book-Author* untuk membentuk fitur teks yang akan diproses.
   
  
   ``` python
       books_df = books_df.dropna()
       books_df = books_df.drop_duplicates('ISBN')
   ```
   - Menghapus seluruh baris null dan data duplikat
     
  
   ``` python
      books_df = books_df.sample(10000, random_state=52, ignore_index=True)
   ```
   - Mengambil 10000 sampel acak dari dataset agar tidak mmemberati proses komputasi
     
   
   ```python
      fix_books_df = pd.DataFrame({
       'isbn': books_df['ISBN'],
       'book_title': books_df['Book-Title'],
       'book_author': books_df['Book-Author'],
       'publication_year': books_df['Year-Of-Publication'],
       'publisher': books_df['Publisher'],
       'book_rating': books_df['Book-Rating']
      })
   ```  
   - Membuat dataset baru fix_books_df.
  
**TF-IDF**: TF-IDF (Term Frequency-Inverse Document Frequency) adalah teknik untuk mengukur pentingnya kata dalam dokumen relatif terhadap korpus dengan menggabungkan frekuensi kata dalam dokumen (TF) dan kelangkaan kata di seluruh korpus (IDF). Dalam sistem rekomendasi berbasis *content-based filtering*, TF-IDF mengubah metadata buku (judul dan penulis) menjadi vektor numerik, memberikan bobot tinggi pada kata-kata spesifik dan rendah pada kata umum, sehingga memungkinkan perhitungan kesamaan antar buku berdasarkan konten.

   ![Screenshot 2025-06-02 145452](https://github.com/user-attachments/assets/bc8a3bfb-35cb-4970-8cc5-13518acb8bf4)
   - Menggunakan *TfidfVectorizer* dari *scikit-learn* untuk mengubah data teks menjadi matriks *TF-IDF*, sebagaimana dilakukan dalam penelitian Ardiansyah et al. (2023) untuk menghitung bobot kata.
     
**Alasan Data Preparation**:
- Pembersihan data & Pemrosesan Data diperlukan untuk memastikan konsistensi dan menghilangkan nilai yang hilang, seperti yang dilakukan dalam preprocessing pada penelitian Ardiansyah et al. (2023).
- *TF-IDF* memungkinkan representasi numerik dari metadata buku, yang penting untuk perhitungan kesamaan.

## Modeling

Sistem rekomendasi dibangun menggunakan pendekatan *content-based filtering* dengan algoritma *TF-IDF* dan *Cosine Similarity*, sebagaimana diimplementasikan dalam penelitian Ardiansyah et al. (2023). Berikut adalah detailnya:

**Cosine Similarity**: *Cosine Similarity* mengukur kesamaan antara dua vektor dengan menghitung kosinus sudut di antara keduanya, menghasilkan nilai dari -1 (berbeda) hingga 1 (mirip). Dalam sistem rekomendasi, digunakan untuk membandingkan vektor TF-IDF antar buku, membentuk matriks kesamaan, dan merekomendasikan buku-buku dengan skor kesamaan tertinggi berdasarkan metadata, efektif untuk data berdimensi tinggi tanpa mempedulikan panjang dokumen.

1. **Pembuatan Matriks Kesamaan**:
   
   ![Screenshot 2025-06-02 145724](https://github.com/user-attachments/assets/3e164d09-1030-42d3-81a9-36d0dabce995)
   
   - Menghitung matriks *Cosine Similarity* menggunakan *cosine_similarity* dari *scikit-learn* untuk mengukur kesamaan antar buku berdasarkan vektor *TF-IDF*.
     
2. **Fungsi Rekomendasi**:
   - Fungsi `book_recommendation` menerima judul buku sebagai input dan mengembalikan *k* buku yang paling mirip berdasarkan skor *Cosine Similarity*.
   - Prosesnya melibatkan pengambilan skor kesamaan dari matriks *cosine_sim_df*, mengurutkan skor dari tertinggi ke terendah, dan mengembalikan *k* buku teratas (kecuali buku input).

3. **Implementasi**:
   - Hasil pemrosesan data yaitu matriks *TF-IDF* dibuat dari kombinasi *Book-Title* dan *Book-Author* untuk menangkap karakteristik konten buku.
   - *Cosine Similarity* dihitung untuk membandingkan vektor *TF-IDF* antar buku, menghasilkan skor kesamaan seperti 0.358 dalam penelitian Ardiansyah et al. (2023).
   - Contoh output untuk buku *The Unicorn Solution* menghasilkan rekomendasi seperti *Facing the Fire: Experiencing and Expressing Anger Appropriately* oleh John Lee dan *Writ Denied* oleh Lee.

4. **Kelebihan dan Kekurangan**:
   - **Kelebihan**: Sistem ini tidak memerlukan data pengguna, hanya metadata buku, sehingga cocok untuk perpustakaan dengan informasi terbatas. Efektif untuk merekomendasikan buku berdasarkan kesamaan konten, seperti yang ditunjukkan dalam penelitian referensi.
   - **Kekurangan**: Terbatas pada fitur metadata yang tersedia (judul dan penulis), sehingga tidak dapat menangkap preferensi pengguna yang lebih kompleks.

**Top-N Recommendation**:

Untuk buku *The Unicorn Solution*, sistem merekomendasikan:

![Screenshot 2025-06-05 120839](https://github.com/user-attachments/assets/8d1634dd-e773-4caa-bff9-ae0dd1701ca7)

## Evaluation

## Metrik Evaluasi
Metrik yang digunakan adalah *precision*, *recall*, dan *F1-score*, dihitung berdasarkan matriks *Cosine Similarity* dengan ambang batas 0.5. Metrik ini dipilih karena sesuai untuk mengevaluasi relevansi rekomendasi, meskipun penelitian Ardiansyah et al. (2023) tidak menyebutkan metrik evaluasi spesifik selain skor *Cosine Similarity*.

## 1. **Precision (Presisi)**
*Precision* mengukur proporsi rekomendasi yang relevan dari semua rekomendasi yang diberikan oleh sistem. Dalam konteks ini, "relevan" berarti buku yang direkomendasikan memiliki skor kesamaan kosinus di atas *threshold* (0.5) sesuai dengan *ground truth*.

- **Rumus**:
  
  $$\text{Precision}=\frac{\text{TP}}{\text{TP}+\text{FP}}$$
  
  - *True Positives (TP)*: Jumlah pasangan buku yang diprediksi mirip (skor ≥ 0.5) dan memang mirip menurut *ground truth*.
  - *False Positives (FP)*: Jumlah pasangan buku yang diprediksi mirip, tetapi sebenarnya tidak mirip menurut *ground truth*.

- **Penjelasan**: Dalam evaluasi, *precision* se personally 1.0, artinya semua buku yang direkomendasikan sebagai "mirip" memang benar-benar mirip menurut *ground truth*. Tidak ada rekomendasi yang salah (*false positive*).

## 2. **Recall (Recall)**
*Recall* mengukur proporsi buku yang benar-benar relevan (mirip) yang berhasil direkomendasikan oleh sistem dari semua buku yang seharusnya relevan menurut *ground truth*.

- **Rumus**:

  $$\text{Recall}=\frac{\text{TP}}{\text{TP}+\text{FP}}$$

  - *True Positives (TP)*: Seperti di atas, jumlah pasangan buku yang diprediksi mirip dan memang mirip.
  - *False Negatives (FN)*: Jumlah pasangan buku yang sebenarnya mirip menurut *ground truth*, tetapi tidak diprediksi mirip oleh sistem (skor < 0.5).

- **Penjelasan**: Nilai *recall* 1.0 menunjukkan bahwa sistem berhasil menangkap semua pasangan buku yang seharusnya mirip menurut *ground truth*. Tidak ada pasangan yang relevan yang terlewat (*false negative*).

## 3. **F1-Score**
*F1-score* adalah rata-rata harmonik dari *precision* dan *recall*, memberikan keseimbangan antara kedua metrik tersebut. Metrik ini berguna untuk mengevaluasi performa model secara keseluruhan, terutama jika *precision* dan *recall* memiliki nilai yang tidak seimbang.

- **Rumus**:
  
  $$\text{F1-Score}=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

- **Penjelasan**: Dengan *precision* dan *recall* masing-masing 1.0, *F1-score* juga bernilai 1.0. Ini menunjukkan performa model yang sempurna dalam evaluasi ini, karena tidak ada kesalahan dalam mengidentifikasi buku yang mirip.

### Hasil Evaluasi

![Screenshot 2025-06-02 150406](https://github.com/user-attachments/assets/24fc147f-85de-4f95-ad92-754a89d641d6)

Hasil evaluasi menunjukkan:
- **Precision**: 1.0
- **Recall**: 1.0
- **F1-Score**: 1.0

**Penjelasan**:
- Nilai *precision*, *recall*, dan *F1-score* yang sempurna (1.0) kemungkinan disebabkan oleh ambang batas 0.5 yang menghasilkan prediksi biner yang sesuai dengan *ground truth*. Namun, seperti yang ditunjukkan dalam penelitian Ardiansyah et al. (2023), skor *Cosine Similarity* (misalnya, 0.358) menunjukkan tingkat kesamaan yang realistis, sehingga hasil evaluasi ini mungkin terlalu optimistis.
- Untuk meningkatkan evaluasi, disarankan untuk menggunakan dataset uji yang lebih beragam atau menyesuaikan ambang batas agar mencerminkan variasi kesamaan yang lebih realistis.

**Hasil & Kesimpulan**:
- Berhasil membangun sistem rekomendasi berbasis *content-based filtering* berhasil memberikan rekomendasi buku yang relevan berdasarkan metadata, dengan skor *Cosine Similarity* yang menunjukkan kesamaan antar buku..
- Hasil evaluasi metrik yang sempurna menunjukkan perlunya pengujian lebih lanjut dengan data yang lebih kompleks untuk memastikan robustitas sistem.

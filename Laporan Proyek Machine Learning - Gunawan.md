# Laporan Proyek Machine Learning - Gunawan

## Domain Proyek
Dalam industri e-commerce yang sangat kompetitif, tingkat churn pelanggan menjadi salah satu tantangan utama yang dapat memengaruhi profitabilitas dan keberlanjutan bisnis. Pelanggan yang churn, atau berhenti menggunakan layanan, sering kali sulit untuk diidentifikasi secara dini, sehingga perusahaan kehilangan peluang untuk mempertahankan mereka. Biaya untuk mendapatkan pelanggan baru biasanya jauh lebih tinggi dibandingkan mempertahankan pelanggan yang sudah ada, sehingga memahami penyebab dan memprediksi churn menjadi krusial ([Early Churn Prediction from Large Scale User-Product Interaction Time Series](https://arxiv.org/abs/2309.14390)).
Proyek ini bertujuan untuk mengidentifikasi pola perilaku pelanggan yang berisiko meninggalkan platform e-commerce dengan memanfaatkan data pelanggan, seperti riwayat pembelian, pola interaksi, dan demografi, untuk mengidentifikasi faktor-faktor utama yang memengaruhi churn. Dengan menerapkan teknik machine learning seperti model prediksi berbasis data, hasil dari analisis ini dapat menyediakan wawasan yang dapat membantu perusahaan dalam merancang strategi retensi pelanggan yang lebih efektif, seperti personalisasi layanan, program loyalitas, atau intervensi proaktif untuk mempertahankan pelanggan yang berisiko churn.
Dengan demikian, hal ini melibatkan pemahaman mendalam tentang pengelolaan data, penerapan algoritme prediksi, dan penerjemahan hasil analisis ke dalam tindakan nyata untuk mengurangi tingkat churn dan meningkatkan kepuasan pelanggan.

## Business Understanding

### Problem Statements
Churn pelanggan dapat menyebabkan kerugian pendapatan yang signifikan serta peningkatan biaya untuk mendapatkan pelanggan baru. Masalah ini penting karena mencegah churn lebih hemat biaya dibandingkan dengan mendapatkan pelanggan baru. Prediksi churn yang dilakukan lebih awal memungkinkan perusahaan untuk secara proaktif berinteraksi dengan pelanggan yang berisiko, meningkatkan kepuasan pelanggan, dan profitabilitas jangka panjang. Tujuannya adalah mengidentifikasi pelanggan yang berisiko churn menggunakan model prediksi berbasis data. Dengan cara ini, perusahaan dapat mengalokasikan sumber daya secara efisien dan memberikan penawaran promosi yang dipersonalisasi untuk mempertahankan pelanggan yang berisiko tinggi.
- Spesifik: Mengidentifikasi pelanggan dengan kemungkinan tinggi untuk churn.
- Terukur: Memprediksi churn dengan tingkat precision dan recall yang tinggi, memaksimalkan F1-Score.
- Dapat Dicapai: Menggunakan data historis pelanggan, termasuk pola perilaku dan transaksi, untuk melatih model prediksi.

### Goals
- Memprediksi kemungkinan churn untuk setiap pelanggan.
- Mengidentifikasi faktor utama yang menyebabkan churn pelanggan untuk memberikan wawasan yang dapat ditindaklanjuti.

### Solution Statements
- Mengembangkan model untuk mengklasifikasikan pelanggan menjadi 2 (dua) kelompok: 0 - Pelanggan tidak berisiko churn, dan 1 - Pelanggan berisiko churn, dengan membandingkan beberapa algoritme prediksi seperti regresi logistik, K-Nearest Neighbors (KNN), pohon keputusan, random forest, XGBoost, dan LightGBM. Kemudian, akan dipilih model yang optimal untuk memecahkan masalah tersebut. 
- Selain itu, pengoptimalan model bisa juga dilakukan melalui hyperparameter tuning. Ini penting untuk menemukan kombinasi parameter terbaik yang dapat meningkatkan performa model.

Metrik evaluasi menggunakan F1-Score, karena:
- Keseimbangan Precision dan Recall: Sebagai harmonic mean dari precision dan recall, F1-Score cocok untuk memastikan model tidak hanya fokus pada meminimalkan false positives (precision) atau false negatives (recall).
- Efektif untuk Data Tidak Seimbang: Dalam kasus churn, data sering kali tidak seimbang (lebih banyak pelanggan tidak churn dibanding churn). F1-Score memberikan gambaran yang lebih realistis dibandingkan akurasi.
Relevan untuk Tujuan Bisnis: Memastikan pelanggan yang benar-benar berisiko churn terdeteksi dengan baik tanpa terlalu banyak kesalahan positif.

## Data Understanding
Dataset e-commerce yang digunakan pada proyek ini berisi informasi tingkat pelanggan, termasuk demografi, riwayat transaksi, metrik perilaku, dan indikator kepuasan. Setiap baris merepresentasikan satu pelanggan, sedangkan setiap kolom menangkap fitur spesifik dari profil mereka. Terdapat sebanyak 5.630 baris dan 20 kolom ([Kaggle - Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)).
- Dataset mencakup informasi perilaku dan demografis, yang penting untuk memahami churn.
- Beberapa fitur memiliki nilai hilang (contoh: Tenure, HourSpendOnApp) yang memerlukan imputasi.
- Data ini cocok untuk teknik pembelajaran terawasi (supervised learning), mengingat adanya variabel target yang jelas (Churn).

### Variabel-variabel pada dataset e-commerce tersebut sebagai berikut:
- CustomerID (Integer): ID pelanggan yang unik
- Churn (Integer): Indikator churn
- Tenure (Float): Lama pelanggan di organisasi
- PreferredLoginDevice (Object): Perangkat login yang paling sering digunakan oleh pelanggan
- CityTier (Integer): Tingkatan kota
- WarehouseToHome (Float): Jarak antara gudang dan rumah pelanggan
- PreferredPaymentMode (Object): Metode pembayaran yang paling sering digunakan oleh pelanggan
- Gender (Object): Jenis kelamin pelanggan
- HourSpendOnApp (Float): Jumlah jam yang dihabiskan pada aplikasi atau situs web
- NumberOfDeviceRegistered (Integer): Total perangkat yang terdaftar atas nama pelanggan
- PreferedOrderCat (Object): Kategori pesanan yang paling sering dipesan pelanggan pada bulan lalu
- SatisfactionScore (Integer): Skor kepuasan pelanggan terhadap layanan
- MaritalStatus (Object): Status pernikahan pelanggan
- NumberOfAddress (Integer): Total alamat yang ditambahkan oleh pelanggan
- Complain (Integer): Ada atau tidaknya komplain dalam bulan terakhir
- OrderAmountHikeFromLastYear (Float): Persentase peningkatan pesanan dibandingkan tahun lalu
- CouponUsed (Float): Total kupon yang digunakan dalam bulan terakhir
- OrderCount (Float): Total jumlah pesanan yang dilakukan dalam bulan terakhir
- DaySinceLastOrder (Float): Hari sejak pesanan terakhir oleh pelanggan
- CashbackAmount (Float): Rata-rata cashback yang diterima dalam bulan terakhir

### Exploratory Data Analysis (EDA)
EDA adalah langkah kritis dalam analisis data yang memungkinkan kita untuk memahami informasi yang terkandung dalam dataset sebelum memulai proses analisis yang lebih mendalam. EDA membantu mengungkap pola, anomali, dan tren yang mungkin tersembunyi dalam data, sehingga memungkinkan pengambilan keputusan yang lebih baik. Dengan menjalankan EDA, kita dapat mengidentifikasi variabel penting, merumuskan pertanyaan penelitian yang lebih tepat, dan membuat asumsi awal yang relevan untuk perancangan model analisis data yang lebih kompleks.

#### Distribusi Data
Untuk melihat distribusi data dari kolom numerik dapat digunakan boxplot. Untuk mempermudah analisis, kolom dibagi menjadi kolom numerikal dan kolom kategorikal. Walaupun Churn, CityTier, dan Complain bertipe data integer, ketiga kolom ini akan masuk ke dalam klasifikasi kategorikal karena kolom tersebut mewakili kategori atau flagging.
![Gambar 01](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar%2001.png)
Berdasarkan boxplot di atas, dapat dilihat bahwa hampir seluruh kolom numerik, kecuali SatisfactionScore, memiliki nilai outlier. Hal ini dapat dilihat dari adanya titik-titik yang berada di luar whisker dan ini menunjukkan adanya nilai ekstrem yang perlu dilihat lebih lanjut.

#### Korelasi Data

Fitur dengan dampak terbesar terhadap churn adalah Tenure dengan korelasi negatif yang signifikan, menunjukkan bahwa loyalitas pelanggan meningkat dengan waktu. DaySinceLastOrder dan CashbackAmount juga memiliki efek negatif moderat terhadap churn, yang berarti menjaga pelanggan aktif dengan cashback dapat membantu mengurangi churn. Fitur lain menunjukkan korelasi yang lemah atau tidak signifikan dengan churn.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

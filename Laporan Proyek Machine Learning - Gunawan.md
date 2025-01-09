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
- Relevan untuk Tujuan Bisnis: Memastikan pelanggan yang benar-benar berisiko churn terdeteksi dengan baik tanpa terlalu banyak kesalahan positif.

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

![Gambar01](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar01.jpg)

Berdasarkan boxplot di atas, dapat dilihat bahwa hampir seluruh kolom numerik, kecuali SatisfactionScore, memiliki nilai outlier. Hal ini dapat dilihat dari adanya titik-titik yang berada di luar whisker dan ini menunjukkan adanya nilai ekstrem yang perlu dilihat lebih lanjut.

![Gambar02](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar02.jpg)

1. **Distribusi dari Tenure:**
  - Distribusi cenderung menurun. Sebagian besar pelanggan memiliki tenure yang pendek (0-10 bulan), sedangkan pelanggan dengan tenure lebih lama menjadi semakin sedikit.
2. **Distribusi dari WarehouseToHome:**
  - Data ini menunjukkan jarak (dalam satuan tertentu) antara gudang ke rumah pelanggan. Sebagian besar jarak antara gudang dan rumah berkisar dalam jarak pendek (0-20), dan jarak lebih jauh menjadi lebih jarang.
3. **Distribusi dari HourSpendOnApp:**
  - Distribusi berbentuk unimodal, dengan puncak utama di sekitar 3 jam. Sebagian besar pelanggan menghabiskan waktu sekitar 2-4 jam di aplikasi.
4. **Distribusi dari NumberOfDeviceRegistered:**
  - Data ini menunjukkan jumlah perangkat yang terdaftar, dengan puncak signifikan pada angka bulat (3, 4, dan 5). Sebagian besar pelanggan tampaknya mendaftarkan 3 hingga 5 perangkat.
5. **Distribusi dari SatisfactionScore:**
  - Distribusi ini hampir normal, dengan mayoritas pelanggan memberikan skor kepuasan sekitar 3 hingga 4, menunjukkan tingkat kepuasan yang cukup baik.
6. **Distribusi dari NumberOfAddress:**
  - Sebagian besar pelanggan memiliki kurang dari 5 alamat yang terdaftar. Distribusi sangat tidak merata, dengan lebih sedikit pelanggan memiliki lebih banyak alamat.
7. **Distribusi dari OrderAmountHikeFromLastYear:**
  - Distribusi ini menunjukkan banyak pelanggan mengalami kenaikan jumlah pesanan di kisaran 12-20 dari tahun sebelumnya. Namun, kenaikan di atas angka tersebut menjadi semakin langka.
8. **Distribusi dari CouponUsed:**
  - Sebagian besar pelanggan tidak menggunakan hingga menggunakan sekitar 2 kupon, tetapi ada pelanggan tertentu yang menggunakan lebih banyak kupon.
9. **Distribution of OrderCount:**
  - Sebagian besar pelanggan melakukan 1-2 pesanan. Frekuensi pelanggan dengan jumlah pesanan lebih tinggi menurun drastis.
10. **Distribution of DaySinceLastOrder:**
   - Distribusi menunjukkan sebagian besar pelanggan baru saja melakukan pesanan terakhirnya (0-10 hari yang lalu), dengan frekuensi yang turun secara eksponensial untuk hari-hari yang lebih lama.
11. **Distribution of CashbackAmount:**
   - Cashback yang diberikan sebagian besar berkisar antara 100-200. Distribusi ini relatif simetris, dengan penyebaran data yang lebih kecil di kedua sisi rata-rata. 

#### Korelasi Data

![Gambar03](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar03.jpg)

Fitur dengan dampak terbesar terhadap churn adalah Tenure dengan korelasi negatif yang signifikan, menunjukkan bahwa loyalitas pelanggan meningkat dengan waktu. DaySinceLastOrder dan CashbackAmount juga memiliki efek negatif moderat terhadap churn, yang berarti menjaga pelanggan aktif dengan cashback dapat membantu mengurangi churn. Fitur lain menunjukkan korelasi yang lemah atau tidak signifikan dengan churn.
Variabel Tenure memiliki korelasi negatif yang cukup signifikan (-0.35) dengan churn, menunjukkan bahwa pelanggan dengan masa keanggotaan lebih lama cenderung lebih kecil kemungkinannya untuk churn. Selain itu, DaySinceLastOrder juga memiliki korelasi negatif moderat (-0.16), yang mengindikasikan bahwa pelanggan yang melakukan pesanan baru-baru ini cenderung tidak churn. Sebaliknya, fitur seperti SatisfactionScore, NumberOfDeviceRegistered, dan CashbackAmount memiliki korelasi positif kecil dengan churn (0.11), namun dampaknya relatif lebih lemah.

#### Kordinalitas Data

![Gambar04](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar04.jpg)

Berdasarkan pengecekan nilai unik pada dataset, pada beberapa kolom, ditemukan beberapa nilai dengan penamaan yang tidak konsisten. Misalnya pada kolom PrefferedLoginDevice terdapat nilai 'Mobile Phone' dan 'Phone' yang merujuk ke perangkat yang sama. Kemudian pada kolom PreferredPaymentMode terdapat 'CC' dan 'Credit Card' yang merujuk kepada penggunaan kartu kredit, serta 'COD' dan 'Cash on Delivery' yang merujuk pada satu metode pembayaran yang sama. Selain itu, pada kolom PreferedOrderCat terdapat 'Mobile' dan 'Mobile Phone' yang merujuk pada kategori yang sama.

#### Identifikasi Nilai Hilang

![Gambar05](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar05.jpg)

Terdapat 7 (tujuh) kolom yang memiliki nilai hilang, yaitu Tenure, WerehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, dan DaySinceLastOrder.

Nilai hilang pada dataset ini dapat diklasifikasikan menjadi 3 (tiga), yaitu: 
1. Missing Completely at Random (MCAR)
  - Data MCAR adalah data yang nilai hilangnya tidak memiliki pola tertentu. Fitur WarehouseToHome dan HourSpendOnApp mungkin termasuk dalam kategori MCAR, karena waktu yang dihabiskan di aplikasi atau jarak ke gudang mungkin tidak bergantung pada fitur lainnya.
  - Pendekatan Penanganan: Karena data MCAR tidak bergantung pada fitur lain, nilai yang hilang dapat diisi dengan nilai rata-rata atau median tanpa menimbulkan bias yang berarti.
2. Missing at Random (MAR)
  - Nilai hilang pada MAR berkaitan dengan data lain yang teramati. Fitur seperti Tenure dan OrderAmountHikeFromlastYear bisa termasuk MAR, karena pelanggan dengan masa penggunaan yang lebih singkat mungkin memiliki lebih banyak data yang hilang, dan kebiasaan belanja (sehingga kenaikan jumlah order) dapat berkorelasi dengan masa penggunaan atau waktu penggunaan aplikasi.
  - Pendekatan Penanganan: Untuk data MAR, metode seperti imputasi berdasarkan fitur yang terkait (misalnya menggunakan pengelompokan berdasarkan masa penggunaan untuk mengisi nilai hilang) atau model prediktif dapat digunakan untuk mengisi nilai hilang berdasarkan pola dalam data yang teramati.
3. Missing Not at Random (MNAR)
  - MNAR terjadi ketika nilai hilang terkait dengan nilai dari fitur itu sendiri. Fitur CouponUsed, OrderCount, dan DaySinceLastOrder mungkin termasuk MNAR jika, misalnya, pelanggan yang jarang menggunakan kupon atau jarang memesan memiliki nilai hilang pada fitur-fitur ini, atau ada periode tanpa pesanan baru.
  - Pendekatan Penanganan: Untuk MNAR, strategi yang efektif meliputi membuat variabel indikator untuk menandai tempat data hilang, karena hilangnya nilai ini bisa saja memiliki informasi penting. Alternatifnya, bisa dipertimbangkan imputasi berdasarkan pengetahuan spesifik domain atau memodelkan missingness langsung jika memungkinkan.

Untuk menangani nilai hilang:
- MCAR: Gunakan imputasi median.
- MAR: Gunakan KNN Imputer.
- MNAR: Buat variabel indikator dan gunakan pengetahuan spesifik domain atau pertimbangkan strategi imputasi lanjutan seperti iterative imputer.

#### Identifikasi Duplikasi Nilai

![Gambar06](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar06.jpg)

Dapat dilihat bahwa tidak terdapat data duplikat yang teridentifikasi.

## Data Preparation

### Nilai Tidak Konsisten

Pada beberapa kolom ditemukan beberapa nilai yang tidak konsisten. Misalnya, pada kolom PrefferedLoginDevice terdapat nilai 'Mobile Phone' dan 'Phone' yang merujuk ke perangkat yang sama. Oleh karena itu, dilakukan penggantian nilai untuk menangani hal ini. Selain kolom PrefferedLoginDevice, pada kolom PreferredPaymentMode dan PreferedOrderCat juga ditemukan hal yang sama.

![Gambar07](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar07.jpg)

### Nilai Hilang

Berdasarkan identifikasi missing value sebelumnya, penanganan missing value untuk kolom Tenure dan OrderAmountHikeFromLastYear akan ditangani dengan menggunakan KNNImputer. Sedangkan pada kolom WareHouseToHome dan HourSpendOnApp akan menggunakan nilai median dari masing-masing kolom.

![Gambar08](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar08.jpg)

Penanganan untuk MNAR: 
1. CouponUsed  
  - Nilai yang hilang pada CouponUsed dapat terjadi jika pelanggan jarang atau bahkan tidak pernah menggunakan kupon. Pendekatan yang dapat dilakukan adalah mengisi nilai hilang dengan nilai 0 (untuk menandakan bahwa kupon tidak digunakan), atau membuat variabel indikator yang mencatat nilai hilang sebagai kategori khusus, misalnya “Tidak Digunakan”.
  - Penggunaan kupon umumnya bersifat sporadis atau preferensi tertentu, sehingga asumsi ini logis.
2. OrderCount  
  - Pada OrderCount, nilai hilang dapat diasumsikan sebagai tidak adanya pesanan pada periode tertentu. Pendekatan ini dapat diatasi dengan mengisi nilai hilang dengan angka 0 atau membuat indikator khusus untuk menunjukkan adanya periode nonpesanan.
  - Banyak e-commerce mencatat periode tanpa aktivitas pesanan, sehingga nilai 0 atau variabel indikator dapat memberikan wawasan lebih lanjut.

![Gambar09](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar09.jpg)

3. DaySinceLastOrder  
  - Jika nilai hilang di kolom DaySinceLastOrder menggambarkan ketidakaktifan atau ketiadaan pesanan, maka mengisi nilai hilang dengan perkiraan tertentu bisa menyesatkan. Menambahkan variabel indikator (flag) untuk menunjukkan pelanggan yang memiliki nilai hilang di kolom ini dapat memberikan insight tambahan. Ini sangat berguna karena memungkinkan analisis terpisah antara pelanggan yang aktif dan tidak aktif.
  - Tambahkan kolom baru, misalnya NoLastOrderInfo, yang berisi nilai 1 jika DaySinceLastOrder hilang dan 0 jika tidak. Setelah menandai, nilai hilang di DaySinceLastOrder bisa diisi dengan nilai median atau mean.
  - Jika pelanggan tertentu tidak pernah memesan, maka isi nilai hilangnya dengan median dan berikan indikator 1 pada kolom NoLastOrderInfo. Dengan demikian, kita dapat mensegmentasi pelanggan berdasarkan aktivitas mereka.
  - Dengan pendekatan ini, kita menggunakan konteks operasional dari e-commerce untuk memastikan bahwa pengisian nilai hilang tetap memberikan informasi yang relevan dan tidak mengaburkan pola dalam data.

### Outlier

Berdasarkan hasil identifikasi sebelumnya, data yang termasuk ke dalam outlier dapat dikatakan cukup banyak sehingga penghapusan memiliki potentsi untuk memengaruhi hasil analisis. Pendekatan penanganan outlier yang dilakukan adalah mengganti nilai outlier dengan nilai ambang batas yang dihitung berdasarkan upper range dan lower range dari masing-masing kolom. Hal ini dilakukan untuk mengurangi pengaruh nilai ekstrim tersebut terhadap model atau analisis tanpa menghapus data.

![Gambar11](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar11.jpg)

### Menghapus Data yang Tidak Dibutuhkan

Berikutnya dapat dilakukan penghapusan kolom yang tidak digunakan untuk analisis, yaitu Customer ID. Kolom CustomerID dapat dihapus karena berisi ID unik pelanggan, dimana ID ini tidak memberikan informasi langsung tentang perilaku atau faktor churn.

![Gambar12](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar12.jpg)

### Encoding

Teknik rekayasa fitur seperti one-hot encoding dan standarisasi sangat penting dalam machine learning karena banyak algoritme tidak dapat menangani data kategorikal atau fitur yang tidak distandarisasi dengan baik (misalnya banyak algoritme seperti regresi linier dan jaringan saraf mengasumsikan bahwa fitur input bersifat numerik dan distandarisasi).

![Gambar13](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar13.jpg)

`ColumnTransformer` adalah alat yang sangat berguna dari scikit-learn yang menerapkan transformasi yang berbeda pada subset kolom yang berbeda dalam data. Fungsi ini mengambil daftar operasi yang akan diterapkan pada kolom-kolom tertentu:
- **`onehot`:** Menerapkan `OneHotEncoder` pada kolom `PreferredLoginDevice`, `Gender`, dan `MaritalStatus`, yang akan mengubah variabel kategorikal menjadi format yang cocok untuk algoritme machine learning (mengubah kategori menjadi format biner).
- **`binary`:** Menerapkan `BinaryEncoder` pada kolom `PreferredPaymentMode` dan `PreferredOrderCat`. Binary encoding adalah alternatif dari one-hot encoding untuk menangani data kategorikal.
- **`num`:** Menerapkan `StandardScaler` pada kolom numerik seperti `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, dan lain-lain. Scaler ini menstandarisasi fitur numerik (mengubahnya agar rata-rata 0 dan variansinya menjadi 1).

## Modeling & Evaluation

### Benchmark Model: K-Fold

Akan diuji cross validation 6 jenis model (regresi logistik, K-Nearest Neighbors (KNN), pohon keputusan, random forest, XGBoost, dan LightGBM) dengan parameter default dan jumlah fold 5 untuk melihat model yang terbaik untuk dataset ini.
1. Logistic Regression (Regresi Logistik):
  - Kelebihan:
    - Cepat dan sederhana untuk digunakan.
    - Hasilnya mudah diinterpretasikan dalam hal probabilitas kelas.
    - Cocok untuk masalah klasifikasi biner atau multikelas.
  - Kekurangan:
    - Tidak efektif untuk hubungan nonlinear antara fitur dan target.
    - Mungkin kurang efektif pada dataset yang sangat besar dengan banyak fitur.
2. K-Nearest Neighbors (KNN):
  - Kelebihan:
    - Mudah dipahami dan diimplementasikan.
    - Tidak memerlukan model pelatihan eksplisit.
    - Sangat efektif pada data dengan batas keputusan yang sederhana.
  - Kekurangan:
    - Lambat saat prediksi pada dataset besar karena harus menghitung jarak ke semua data setiap kali prediksi dilakukan.
    - Kinerja dapat menurun jika data memiliki banyak fitur (dimensionality curse).
3. Decision Tree (Pohon Keputusan):
  - Kelebihan:
    - Mudah dipahami dan interpretasi model langsung.
    - Bisa menangani data numerik dan kategorikal.
    - Dapat menangani hubungan nonlinear.
  - Kekurangan:
    - Rentan terhadap overfitting, terutama jika pohon terlalu dalam.
    - Tidak selalu memberikan hasil yang stabil, terutama jika data noisy.
4. Random Forest:
  - Kelebihan:
    - Meningkatkan akurasi dengan mengurangi risiko overfitting yang ada pada decision tree tunggal.
    - Lebih stabil terhadap fluktuasi data dan noise.
    - Dapat menangani data yang hilang dan outlier.
  - Kekurangan:
    - Model yang lebih kompleks dan memerlukan lebih banyak sumber daya komputasi.
    - Kurang interpretable dibandingkan pohon keputusan tunggal.
5. XGBoost (Extreme Gradient Boosting):
  - Kelebihan:
    - Kinerja sangat baik untuk banyak masalah klasifikasi dan regresi.
    - Mampu menangani data yang tidak terstruktur dan besar dengan efisien.
    - Menghasilkan model yang lebih baik dengan waktu pelatihan yang lebih cepat dibandingkan dengan gradient boosting tradisional.
  - Kekurangan:
    - Lebih rumit untuk diatur dibandingkan dengan Random Forest atau Decision Tree.
    - Memerlukan tuning parameter yang lebih hati-hati untuk mencapai performa optimal.
6. LightGBM (Light Gradient Boosting Machine):
 - Kelebihan:
    - Sangat cepat dan efisien dalam hal memori.
    - Dapat menangani dataset besar dengan jumlah fitur yang sangat banyak.
    - Sering kali menghasilkan model yang lebih baik dengan lebih sedikit waktu pelatihan dibandingkan dengan XGBoost.
  - Kekurangan:
    - Tuning parameter bisa cukup rumit.
    - Mungkin kurang stabil pada dataset yang lebih kecil atau sangat terstruktur dibandingkan XGBoost.

Cross-validation adalah teknik yang sangat penting untuk menguji kinerja model secara lebih robust. Dengan membagi data ke dalam beberapa lipatan dan melatih model pada sebagian data, kemudian menguji pada bagian yang lain, kita mendapatkan gambaran yang lebih baik tentang bagaimana model akan bekerja di data yang belum pernah dilihat sebelumnya.

![Gambar14](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar14.jpg)

Setelah cross-validation, hasil evaluasi untuk masing-masing model (`Logistic Regression`, `KNN`, `Decision Tree`, `Random Forest`, `XGBoost`, dan `LightGBM`) ditampilkan. Hasilnya adalah DataFrame yang menampilkan F1-macro score untuk setiap model, yang digunakan untuk menentukan model terbaik berdasarkan kinerja mereka.

### Benchmark Model: Test Data

![Gambar15](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar15.jpg)

Setelah menjalankan kode ini, hasilnya adalah tabel yang menunjukkan evaluasi dari berbagai model klasifikasi berdasarkan metrik, seperti **precision**, **recall**, **f1-score**, **accuracy**, dan **weighted avg** untuk setiap model yang diuji. Tabel yang ditampilkan mengurutkan model berdasarkan skor **F1-macro**, sehingga model dengan performa terbaik (berdasarkan F1-macro) akan berada di atas. **XGBoost** memiliki F1-macro tertinggi (0.960177), diikuti oleh **Random Forest**, **Decision Tree**, **LightGBM**, dan **Logistic Regression**.

![Gambar16](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar16.jpg)

### Pengujian Oversampling dengan K-Fold Cross Validation

Ketika dataset tidak seimbang (misalnya ada lebih banyak contoh dari satu kelas dibandingkan kelas lainnya), Random Oversampling digunakan untuk menambah jumlah data dari kelas minoritas. Ini membantu model untuk belajar lebih baik pada kelas yang lebih sedikit dan mengurangi bias terhadap kelas mayoritas.
Dengan membandingkan hasil dari model yang dilatih dengan dan tanpa oversampling, kita dapat mengetahui apakah oversampling memberi dampak positif terhadap performa model dalam hal metrik evaluasi, seperti F1-score, AUC, precision, dan recall.
Train Errors (Kesalahan Pelatihan) dan Validation Errors (Kesalahan Validasi) dihitung untuk masing-masing fold dan dibandingkan antara model yang menggunakan oversampling dan yang tidak

#### Metrik Evaluasi Tanpa Oversampling

![Gambar17](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar17.jpg)

#### Metrik Evaluasi Dengan Oversampling

![Gambar18](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar18.jpg)

Hasil yang ditampilkan adalah DataFrame yang berisi metrik evaluasi untuk setiap fold, serta rata-rata metrik untuk semua fold. DataFrame ini mencakup:
- Train Accuracy: Akurasi model pada data pelatihan.
- Test Accuracy: Akurasi model pada data uji.
- Train ROC AUC: Nilai ROC AUC untuk data pelatihan.
- Test ROC AUC: Nilai ROC AUC untuk data uji.
- Train F1-Score: F1-Score untuk data pelatihan.
- Test F1-Score: F1-Score untuk data uji.
- Train Recall: Recall pada data pelatihan.
- Test Recall: Recall pada data uji.
- Train Precision: Precision pada data pelatihan.
- Test Precision: Precision pada data uji.

Hasilnya adalah sebagai berikut.
- Train Accuracy: Nilai akurasi untuk data pelatihan cenderung 1.0, yang menunjukkan bahwa model cukup baik dalam mempelajari data pelatihan.
- Test Accuracy: Nilai akurasi pada data uji berkisar antara 0.95 hingga 1.0, yang menunjukkan bahwa model juga bekerja dengan baik pada data yang tidak terlihat sebelumnya.
- Train ROC AUC dan Test ROC AUC: Nilai AUC yang lebih tinggi menunjukkan kemampuan model dalam membedakan kelas positif dan negatif. Nilai-nilai ini mendekati 1.0, yang menunjukkan model bekerja sangat baik dalam hal ini.
Train F1-Score dan Test F1-Score: F1-score yang lebih tinggi menunjukkan keseimbangan yang baik antara precision dan recall.
- Train Recall dan Test Recall: Nilai recall yang tinggi menunjukkan model dapat mengenali hampir semua contoh dari kelas positif.
- Train Precision dan Test Precision: Precision yang tinggi menunjukkan bahwa sebagian besar prediksi positif model adalah benar.

### Hyperparameter Tuning

RandomizedSearchCV digunakan untuk mencari kombinasi terbaik dari hyperparameter untuk model XGBoost. Pendekatan ini memungkinkan pencarian di berbagai kombinasi hyperparameter tanpa melakukan pencarian menyeluruh (seperti pada GridSearchCV), yang lebih efisien pada dataset besar.

![Gambar19](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar19.jpg)

Di sini, kita mendefinisikan **ruang pencarian untuk hyperparameter** yang akan dicoba selama pencarian acak (**RandomizedSearchCV**).
- **`'n_estimators': [50, 100, 200]`**: Jumlah pohon yang akan dibangun dalam model XGBoost.
- **`'max_depth': [None, 10, 20, 30]`**: Kedalaman maksimum pohon. Kedalaman yang lebih besar memungkinkan model untuk menangkap lebih banyak detail dalam data, tetapi berisiko overfitting.
- **`'min_samples_split': [2, 5, 10]`**: Jumlah sampel minimum yang diperlukan untuk membagi node. Ini membantu dalam mencegah overfitting dengan mengatur seberapa sensitif pembagian pohon.
- **`'max_features': ['sqrt', 'log2', None]`**: Mengontrol jumlah fitur yang dipertimbangkan saat membagi setiap node pohon.
- **`'random_state': [42]`**: Untuk memastikan hasil yang dapat direproduksi selama pencarian acak.

Kombinasi hyperparameter terbaik yang ditemukan setelah pencarian acak kemudian disimpan sebagai model terbaik dan model terbaik dilatih dengan data pelatihan.

![Gambar20](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar20.jpg)

Selanjutnya ditampilkan laporan klasifikasi yang memberikan informasi tentang precision, recall, F1-score, dan support untuk setiap kelas sebelum tuning hyperparameter, serta informasi serupa untuk model XGBoost yang telah di-tuning dengan hyperparameter terbaik yang ditemukan selama pencarian acak.

![Gambar21](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar21.jpg)

Model XGBoost yang telah disesuaikan menunjukkan performa luar biasa dengan **f1-score macro sebesar 0.97**, yang mencerminkan keseimbangan antara presisi dan recall di kedua kelas (churn dan tidak churn). Dengan tingkat akurasi keseluruhan 98%, model ini sangat andal dalam memprediksi pelanggan yang berpotensi churn (kelas 1) maupun yang tetap loyal (kelas 0). Namun, mempertahankan **f1-score** di tingkat ini sangat penting karena secara langsung memengaruhi kemampuan untuk menangani churn pelanggan secara akurat.


## Kesimpulan

1. Pelanggan yang berisiko churn seringkali adalah mereka yang merasa kurang terlibat secara emosional maupun fungsional dengan layanan e-commerce. Mereka cenderung mencari pengalaman yang lebih baik, harga yang lebih kompetitif, atau layanan yang lebih sesuai dengan kebutuhan spesifik mereka. Mereka selalu menjelajahi opsi lain karena merasa layanan saat ini tidak cukup memuaskan, baik dari segi kenyamanan, kecepatan, atau insentif yang mereka harapkan. Mereka cenderung memiliki ekspektasi tinggi, tetapi dengan toleransi rendah terhadap ketidaksesuaian, menjadikan mereka kelompok yang memerlukan perhatian khusus dan pendekatan personal agar tetap setia.
2. Model XGBoost yang telah disesuaikan menunjukkan performa luar biasa dengan **f1-score macro sebesar 0.97**, yang mencerminkan keseimbangan antara presisi dan recall di kedua kelas (churn dan tidak churn). Dengan tingkat akurasi keseluruhan 98%, model ini sangat andal dalam memprediksi pelanggan yang berpotensi churn (kelas 1) maupun yang tetap loyal (kelas 0). Namun, mempertahankan **f1-score** di tingkat ini sangat penting karena secara langsung memengaruhi kemampuan untuk menangani churn pelanggan secara akurat.
3. Penerapan sistem Machine Learning berbasis XGBoost telah memberikan peningkatan signifikan dalam efisiensi prediksi risiko churn, dengan F1-Score sebesar 0.94 pada yang berisiko churn, mencerminkan keseimbangan optimal antara precision (0.94) dan recall (0.94).

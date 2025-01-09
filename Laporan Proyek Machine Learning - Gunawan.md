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

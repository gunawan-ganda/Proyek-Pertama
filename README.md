# Laporan Proyek Machine Learning - Gunawan

## Domain Proyek
Dalam industri e-commerce yang sangat kompetitif, tingkat churn pelanggan menjadi salah satu tantangan utama yang dapat memengaruhi profitabilitas dan keberlanjutan bisnis. Pelanggan yang churn, atau berhenti menggunakan layanan, sering kali sulit untuk diidentifikasi secara dini, sehingga perusahaan kehilangan peluang untuk mempertahankan mereka. Biaya untuk mendapatkan pelanggan baru biasanya jauh lebih tinggi dibandingkan mempertahankan pelanggan yang sudah ada, sehingga memahami penyebab dan memprediksi churn menjadi krusial ([Early Churn Prediction from Large Scale User-Product Interaction Time Series](https://arxiv.org/abs/2309.14390)).

Proyek ini bertujuan untuk mengidentifikasi pola perilaku pelanggan yang berisiko meninggalkan platform e-commerce dengan memanfaatkan data pelanggan, seperti riwayat pembelian, pola interaksi, dan demografi, untuk mengidentifikasi faktor-faktor utama yang memengaruhi churn. Dengan menerapkan teknik machine learning seperti model prediksi berbasis data, hasil dari analisis ini dapat menyediakan wawasan yang dapat membantu perusahaan dalam merancang strategi retensi pelanggan yang lebih efektif, seperti personalisasi layanan, program loyalitas, atau intervensi proaktif untuk mempertahankan pelanggan yang berisiko churn.

Dengan demikian, hal ini melibatkan pemahaman mendalam tentang pengelolaan data, penerapan algoritme prediksi, dan penerjemahan hasil analisis ke dalam tindakan nyata untuk mengurangi tingkat churn dan meningkatkan kepuasan pelanggan.

## Business Understanding

### Problem Statements
Churn pelanggan dapat menyebabkan kerugian pendapatan yang signifikan serta peningkatan biaya untuk mendapatkan pelanggan baru. Masalah ini penting karena mencegah churn lebih hemat biaya dibandingkan dengan mendapatkan pelanggan baru. Prediksi churn yang dilakukan lebih awal memungkinkan perusahaan untuk secara proaktif berinteraksi dengan pelanggan yang berisiko, meningkatkan kepuasan pelanggan, dan profitabilitas jangka panjang. Berdasarkan kondisi yang telah diuraikan sebelumnya, maka pada proyek ini dapat diambil rumusan masalah sebagai berikut.
- Bagaimana mengidentifikasi pelanggan yang berisiko churn menggunakan model prediksi berbasis data?
- Dari serangkaian fitur yang ada, fitur apa saja yang paling berpengaruh terhadap churn pelanggan?

### Goals
Berdasarkan masalah yang telah dirumuskan sebelumnya, maka tujuan dari proyek ini adalah:
- Membuat model machine learning yang dapat memprediksi churn pelanggan seakurat mungkin berdasarkan fitur-fitur yang ada.
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

![Gambar01](https://github.com/user-attachments/assets/edc2dd32-0db1-4b20-99ca-fced2b4a40b4)

Berdasarkan boxplot di atas, dapat dilihat bahwa hampir seluruh kolom numerik, kecuali SatisfactionScore, memiliki nilai outlier. Hal ini dapat dilihat dari adanya titik-titik yang berada di luar whisker dan ini menunjukkan adanya nilai ekstrem yang perlu dilihat lebih lanjut.

![Gambar02](https://github.com/user-attachments/assets/85e49849-3d86-4815-a22b-b2bfc95ce1f9)

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

![Gambar03](https://github.com/user-attachments/assets/869fc714-8f61-4367-bcd9-af364bb8cd08)

Fitur dengan dampak terbesar terhadap churn adalah Tenure dengan korelasi negatif yang signifikan, menunjukkan bahwa loyalitas pelanggan meningkat dengan waktu. DaySinceLastOrder dan CashbackAmount juga memiliki efek negatif moderat terhadap churn, yang berarti menjaga pelanggan aktif dengan cashback dapat membantu mengurangi churn. Fitur lain menunjukkan korelasi yang lemah atau tidak signifikan dengan churn.
Variabel Tenure memiliki korelasi negatif yang cukup signifikan (-0.35) dengan churn, menunjukkan bahwa pelanggan dengan masa keanggotaan lebih lama cenderung lebih kecil kemungkinannya untuk churn. Selain itu, DaySinceLastOrder juga memiliki korelasi negatif moderat (-0.16), yang mengindikasikan bahwa pelanggan yang melakukan pesanan baru-baru ini cenderung tidak churn. Sebaliknya, fitur seperti SatisfactionScore, NumberOfDeviceRegistered, dan CashbackAmount memiliki korelasi positif kecil dengan churn (0.11), namun dampaknya relatif lebih lemah.

#### Kordinalitas Data

![Gambar04](https://github.com/user-attachments/assets/d8fd6315-debe-494a-a955-0e62a39904f8)

Berdasarkan pengecekan nilai unik pada dataset, pada beberapa kolom, ditemukan beberapa nilai dengan penamaan yang tidak konsisten. Misalnya pada kolom PrefferedLoginDevice terdapat nilai 'Mobile Phone' dan 'Phone' yang merujuk ke perangkat yang sama. Kemudian pada kolom PreferredPaymentMode terdapat 'CC' dan 'Credit Card' yang merujuk kepada penggunaan kartu kredit, serta 'COD' dan 'Cash on Delivery' yang merujuk pada satu metode pembayaran yang sama. Selain itu, pada kolom PreferedOrderCat terdapat 'Mobile' dan 'Mobile Phone' yang merujuk pada kategori yang sama.

#### Identifikasi Nilai Hilang

![Gambar05](https://github.com/user-attachments/assets/cec1b0e5-576c-485e-8494-5be776cf570d)

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

![Gambar06](https://github.com/user-attachments/assets/d0c65ba0-86ee-456f-8686-16cacc0aac4e)

Dapat dilihat bahwa tidak terdapat data duplikat yang teridentifikasi.

## Data Preparation

### Nilai Tidak Konsisten

Pada beberapa kolom ditemukan beberapa nilai yang tidak konsisten. Misalnya, pada kolom PrefferedLoginDevice terdapat nilai 'Mobile Phone' dan 'Phone' yang merujuk ke perangkat yang sama. Oleh karena itu, dilakukan penggantian nilai untuk menangani hal ini. Selain kolom PrefferedLoginDevice, pada kolom PreferredPaymentMode dan PreferedOrderCat juga ditemukan hal yang sama.

![Gambar07](https://github.com/user-attachments/assets/4b5109a9-1ba5-4b53-badf-599b297e578f)

### Nilai Hilang

Berdasarkan identifikasi missing value sebelumnya, penanganan missing value untuk kolom Tenure dan OrderAmountHikeFromLastYear akan ditangani dengan menggunakan KNNImputer. Sedangkan pada kolom WareHouseToHome dan HourSpendOnApp akan menggunakan nilai median dari masing-masing kolom.

![Gambar08](https://github.com/user-attachments/assets/1fccf778-b2ad-414d-a396-53cee41bd5c9)

Penanganan untuk MNAR: 
1. CouponUsed  
  - Nilai yang hilang pada CouponUsed dapat terjadi jika pelanggan jarang atau bahkan tidak pernah menggunakan kupon. Pendekatan yang dapat dilakukan adalah mengisi nilai hilang dengan nilai 0 (untuk menandakan bahwa kupon tidak digunakan), atau membuat variabel indikator yang mencatat nilai hilang sebagai kategori khusus, misalnya “Tidak Digunakan”.
  - Penggunaan kupon umumnya bersifat sporadis atau preferensi tertentu, sehingga asumsi ini logis.
2. OrderCount  
  - Pada OrderCount, nilai hilang dapat diasumsikan sebagai tidak adanya pesanan pada periode tertentu. Pendekatan ini dapat diatasi dengan mengisi nilai hilang dengan angka 0 atau membuat indikator khusus untuk menunjukkan adanya periode nonpesanan.
  - Banyak e-commerce mencatat periode tanpa aktivitas pesanan, sehingga nilai 0 atau variabel indikator dapat memberikan wawasan lebih lanjut.

![Gambar09](https://github.com/user-attachments/assets/e4b18be2-c61b-4525-9490-fda8c96b0e11)

3. DaySinceLastOrder  
  - Jika nilai hilang di kolom DaySinceLastOrder menggambarkan ketidakaktifan atau ketiadaan pesanan, maka mengisi nilai hilang dengan perkiraan tertentu bisa menyesatkan. Menambahkan variabel indikator (flag) untuk menunjukkan pelanggan yang memiliki nilai hilang di kolom ini dapat memberikan insight tambahan. Ini sangat berguna karena memungkinkan analisis terpisah antara pelanggan yang aktif dan tidak aktif.
  - Tambahkan kolom baru, misalnya NoLastOrderInfo, yang berisi nilai 1 jika DaySinceLastOrder hilang dan 0 jika tidak. Setelah menandai, nilai hilang di DaySinceLastOrder bisa diisi dengan nilai median atau mean.
  - Jika pelanggan tertentu tidak pernah memesan, maka isi nilai hilangnya dengan median dan berikan indikator 1 pada kolom NoLastOrderInfo. Dengan demikian, kita dapat mensegmentasi pelanggan berdasarkan aktivitas mereka.
  - Dengan pendekatan ini, kita menggunakan konteks operasional dari e-commerce untuk memastikan bahwa pengisian nilai hilang tetap memberikan informasi yang relevan dan tidak mengaburkan pola dalam data.

![Gambar10](https://github.com/user-attachments/assets/92d74f11-cb0a-49d2-8112-cfd165c4029f)

### Outlier

Berdasarkan hasil identifikasi sebelumnya, data yang termasuk ke dalam outlier dapat dikatakan cukup banyak sehingga penghapusan memiliki potentsi untuk memengaruhi hasil analisis. Pendekatan penanganan outlier yang dilakukan adalah mengganti nilai outlier dengan nilai ambang batas yang dihitung berdasarkan upper range dan lower range dari masing-masing kolom. Hal ini dilakukan untuk mengurangi pengaruh nilai ekstrim tersebut terhadap model atau analisis tanpa menghapus data.

![Gambar11](https://github.com/user-attachments/assets/b33a65f8-094b-4b7d-a860-1c08567e68e1)

### Menghapus Data yang Tidak Dibutuhkan

Berikutnya dapat dilakukan penghapusan kolom yang tidak digunakan untuk analisis, yaitu Customer ID. Kolom CustomerID dapat dihapus karena berisi ID unik pelanggan, dimana ID ini tidak memberikan informasi langsung tentang perilaku atau faktor churn.

![Gambar12](https://github.com/user-attachments/assets/4b3c93b9-7177-4bae-a142-36b718f24565)

### Encoding

Teknik rekayasa fitur seperti one-hot encoding dan standarisasi sangat penting dalam machine learning karena banyak algoritme tidak dapat menangani data kategorikal atau fitur yang tidak distandarisasi dengan baik (misalnya banyak algoritme seperti regresi linier dan jaringan saraf mengasumsikan bahwa fitur input bersifat numerik dan distandarisasi).

![Gambar13](https://github.com/user-attachments/assets/65e392e7-6702-42e1-b509-8b7d5163cd71)

`ColumnTransformer` adalah alat yang sangat berguna dari scikit-learn yang menerapkan transformasi yang berbeda pada subset kolom yang berbeda dalam data. Fungsi ini mengambil daftar operasi yang akan diterapkan pada kolom-kolom tertentu:
- **`onehot`:** Menerapkan `OneHotEncoder` pada kolom `PreferredLoginDevice`, `Gender`, dan `MaritalStatus`, yang akan mengubah variabel kategorikal menjadi format yang cocok untuk algoritme machine learning (mengubah kategori menjadi format biner).
- **`binary`:** Menerapkan `BinaryEncoder` pada kolom `PreferredPaymentMode` dan `PreferredOrderCat`. Binary encoding adalah alternatif dari one-hot encoding untuk menangani data kategorikal.
- **`num`:** Menerapkan `StandardScaler` pada kolom numerik seperti `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, dan lain-lain. Scaler ini menstandarisasi fitur numerik (mengubahnya agar rata-rata 0 dan variansinya menjadi 1).

### Pemecahan Data

![Gambar13_1](https://github.com/user-attachments/assets/adcc6c39-2237-4084-b3b7-b27ab4393808)

Variabel `x` berisi semua kolom kecuali kolom target Churn, sementara variabel `y` berisi kolom target Churn. Ini adalah cara yang umum dalam mempersiapkan data untuk melatih model machine learning (dimana `x` adalah fitur dan `y` adalah target).

### Pembagian Data untuk Latihan dan Pengujian

![Gambar13_2](https://github.com/user-attachments/assets/14c2df45-3b24-4d02-940d-ae00a0fb72c6)

Fungsi `train_test_split` digunakan untuk membagi data menjadi set pelatihan dan pengujian. `x_train` dan `x_test` adalah fitur untuk pelatihan dan pengujian, sedangkan `y_train` dan `y_test` adalah target untuk pelatihan dan pengujian. Argumen `stratify=y` memastikan bahwa pembagian data mempertahankan distribusi variabel target (Churn) di kedua set pelatihan dan pengujian. `test_size=0.2` berarti 20% dari data digunakan untuk pengujian.

## Modeling

![Gambar14_1](https://github.com/user-attachments/assets/c2cf9ea6-630b-4193-9420-daaed20438ba)

Pada proyek ini akan diuji 6 jenis model, yaitu regresi logistik, K-Nearest Neighbors (KNN), pohon keputusan, random forest, XGBoost, dan LightGBM. Semua model ini adalah model klasifikasi yang populer dan digunakan untuk membandingkan kinerja berbagai algoritme dalam hal akurasi dan performa lainnya. Dengan membandingkan beberapa model, dapat ditentukan algoritme mana yang paling cocok untuk dataset yang digunakan berdasarkan performa yang diukur sehingga membantu dalam memilih model yang paling optimal untuk klasifikasi yang lebih baik. Parameter yang digunakan pada algoritme regresi logistik adalah `max_iter=2000` dan `random_state=42`. `max_iter` menentukan jumlah iterasi maksimum yang diizinkan untuk algoritme optimisasi saat meminimalkan fungsi biaya. `random_state` mengontrol seeding generator angka acak yang digunakan dalam model. Algoritme-algoritme lainnya selain regresi logistik akan menggunakan **parameter default**.

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

### Benchmark Model: K-Fold

![Gambar14_2](https://github.com/user-attachments/assets/dd027b0a-38a8-4e88-a6b1-7347045e1405)

Cross-validation adalah teknik yang sangat penting untuk menguji kinerja model secara lebih robust. Dengan membagi data ke dalam beberapa lipatan dan melatih model pada sebagian data, kemudian menguji pada bagian yang lain, akan didapatkan gambaran yang lebih baik tentang bagaimana model akan bekerja di data yang belum pernah dilihat sebelumnya.
- StratifiedKFold digunakan untuk membagi data menjadi beberapa "fold" atau lipatan untuk validasi silang (cross-validation). Stratifikasi memastikan bahwa distribusi label target (`y_train`) terjaga di setiap fold.
- `n_splits=5`: Data dibagi menjadi 5 lipatan (folds) untuk cross-validation.
- `shuffle=True`: Data akan diacak sebelum dibagi menjadi lipatan.
- `random_state=42`: Penetapan nilai acak untuk memastikan hasil yang dapat direproduksi.

### Benchmark Model: Test Data

![Gambar15_1](https://github.com/user-attachments/assets/35bd8fab-818b-4d61-89fe-2d6bb3744a05)

- `models = [...]`: Daftar model yang digunakan untuk benchmarking. Model-model tersebut adalah regresi logistik, KNN (K-Nearest Neighbors), pohon keputusan, Random Forest, XGBoost, dan LightGBM.
- `def y_pred_func(...)`: Fungsi ini bertujuan untuk melatih model dan mengembalikan hasil prediksi berdasarkan data tes. Pipeline diterapkan untuk menggabungkan preprocessing dan model pelatihan. Setelah melatih model, fungsi ini akan menghasilkan prediksi menggunakan data uji (`x_test`) dan mengembalikannya.
- `for i, j in zip(models, models_name)`: Looping untuk melakukan prediksi, dimana setiap model dalam daftar `models` dilatih dan diuji pada data uji (`x_test`). Model dilatih dengan data pelatihan (`x_train` dan `y_train`), kemudian hasil prediksinya dihitung menggunakan `model.predict()`.

### Pemilihan Model Terbaik dari Benchmark Model

Berdasarkan eksperimen yang telah dilakukan pada tahapan pengembangan model, diperoleh model machine learning yang berkinerja terbaik, yaitu XGBoost. Hal ini berdasarkan hasil dari skor F1-macro.

### Pengujian Oversampling dengan K-Fold Cross Validation ###

Oversampling adalah teknik yang digunakan dalam pengolahan data tidak seimbang (imbalanced dataset) untuk menangani situasi dimana satu kelas memiliki jauh lebih banyak sampel dibandingkan kelas lainnya. Pengujian oversampling bertujuan untuk meningkatkan kualitas model dengan mengurangi bias yang mungkin timbul akibat ketidakseimbangan kelas. Ketika dataset tidak seimbang (misalnya ada lebih banyak contoh dari satu kelas dibandingkan kelas lainnya), Random Oversampling digunakan untuk menambah jumlah data dari kelas minoritas. Ini membantu model untuk belajar lebih baik pada kelas yang lebih sedikit dan mengurangi bias terhadap kelas mayoritas. Dengan membandingkan hasil dari model yang dilatih dengan dan tanpa oversampling, kita dapat mengetahui apakah oversampling memberi dampak positif terhadap performa model. 

![Gambar17_1](https://github.com/user-attachments/assets/1fb472b3-ef75-4c9d-b619-211b462588db)

- `KFold(n_splits=10, shuffle=True, random_state=42)`: Teknik K-Fold Cross Validation yang membagi data menjadi 10 lipatan (folds).
- `shuffle=True`: Memastikan bahwa data diacak terlebih dahulu sebelum dibagi ke dalam lipatan.
- `random_state=42`: Menetapkan nilai acak untuk memastikan hasil yang dapat direproduksi.
- `ros=RandomOverSampler(random_state=42)`: RandomOverSampler digunakan untuk mengatasi ketidakseimbangan kelas dengan menambahkan sampel ke kelas minoritas (oversampling).
- `X_ros, y_ros=ros.fit_resample(X_train, y_train)`: Menghasilkan data pelatihan baru dengan kelas minoritas yang lebih banyak untuk mengimbangi kelas mayoritas.
- `xgb=XGBClassifier()`: Model XGBoost yang digunakan untuk pengujian dengan dan tanpa oversampling.

Berdasarkan eksperimen yang telah dilakukan dengan dan tanpa oversampling, diperoleh hasil XGBoost dengan oversampling merupakan model machine learning yang berkinerja terbaik. Hal ini berdasarkan hasil dari skor F1.

### Hyperparameter Tuning

Hyperparameter digunakan untuk meng-custom model dan mengontrol proses training sesuai dengan dataset atau permasalahan yang ingin diselesaikan. Sementara hyperparameter tuning dilakukan dengan tujuan untuk memperoleh konfigurasi yang paling optimal untuk melatih model machine learning. Pada praktiknya, proses hyperparameter tuning ini dapat dijalankan secara manual dengan mencoba berbagai konfigurasi hyperparameter yang ada hingga diperoleh konfigurasi yang paling optimal. Cara lainnya yaitu dengan melakukan hyperparameter tuning secara otomatis dengan bantuan beberapa algoritme, salah satunya adalah random search. Pada proyek ini digunakan RandomizedSearchCV untuk mencari kombinasi terbaik dari hyperparameter untuk model XGBoost. Pendekatan ini memungkinkan pencarian di berbagai kombinasi hyperparameter tanpa melakukan pencarian menyeluruh (seperti pada GridSearchCV), yang lebih efisien pada dataset besar.

![Gambar19](https://github.com/user-attachments/assets/f4287b83-cf1e-49a0-b835-ed377de3c554)

**Ruang pencarian untuk hyperparameter** yang akan dicoba selama pencarian acak (**RandomizedSearchCV**) adalah:
- **`'n_estimators': [50, 100, 200]`**: Jumlah pohon yang akan dibangun dalam model XGBoost.
- **`'max_depth': [None, 10, 20, 30]`**: Kedalaman maksimum pohon. Kedalaman yang lebih besar memungkinkan model untuk menangkap lebih banyak detail dalam data, tetapi berisiko overfitting.
- **`'min_samples_split': [2, 5, 10]`**: Jumlah sampel minimum yang diperlukan untuk membagi node. Ini membantu dalam mencegah overfitting dengan mengatur seberapa sensitif pembagian pohon.
- **`'max_features': ['sqrt', 'log2', None]`**: Mengontrol jumlah fitur yang dipertimbangkan saat membagi setiap node pohon.
- **`'random_state': [42]`**: Untuk memastikan hasil yang dapat direproduksi selama pencarian acak.

Kombinasi hyperparameter terbaik yang ditemukan setelah pencarian acak kemudian disimpan sebagai model terbaik dan model terbaik dilatih dengan data pelatihan.

![Gambar20](https://github.com/user-attachments/assets/99dd4ace-12eb-4958-b9e5-2827f79701d6)

### Pemilihan Model Terbaik

Berdasarkan eksperimen yang telah dilakukan pada tahapan-tahapan sebelumnya, diperoleh model machine learning terbaik yaitu XGBoost oversampling dengan hyperparameter tuning random search. Hal ini berdasarkan hasil dari skor F1 macro dan classification report yang menunjukkan bahwa hasil dari model ini lebih baik dari hasil model XGBoost oversampling yang tidak menerapkan hyperparameter tuning, sehingga hasil dari model machine learing yang dikembangkan telah memenuhi tujuan dari solution statement yang telah ditentukan sebelumnya.

### Fitur Penting

Berikutnya dilakukan analisis untuk memvisualisasikan importance (pentingnya) fitur dalam sebuah model machine learning dengan menggunakan atribut `feature_importances_`. Dari hasil visualisasi, terlihat bahwa 3 fitur yang penting dari dataset yang memengaruhi churn pelanggan adalah Tenure, Complain, dan PreferedOrderCat. Hasil ini telah memenuhi tujuan dari solution statement yang telah ditentukan sebelumnya

![Gambar22](https://github.com/user-attachments/assets/fb30f544-2005-40ed-8156-3431ea91ecec)

## Evaluation

F1-Score dipilih sebagai metrik evaluasi utama dalam model ini.
- Mengapa tidak hanya menggunakan precision? Precision tinggi berarti model berhasil mengidentifikasi sebagian besar pelanggan yang benar-benar berisiko churn. Namun, jika precision tinggi tetapi recall rendah, sistem otomatis akan memberikan promo kepada pelanggan yang sebenarnya tidak berisiko churn (False Positives). Hal ini berpotensi meningkatkan biaya operasional secara tidak perlu.
- Mengapa tidak hanya menggunakan recall? Recall tinggi memastikan hanya pelanggan yang benar-benar berisiko churn yang diberi promo. Namun, jika recall tinggi tetapi precision rendah, sistem akan gagal mendeteksi sebagian besar pelanggan yang berisiko churn. Akibatnya, bisnis kehilangan kesempatan mempertahankan pelanggan penting.
- Mengapa F1-Score lebih relevan? F1-Score merupakan harmonic mean dari precision dan recall, memberikan keseimbangan antara kedua metrik tersebut. Dalam sistem seperti ini, keseimbangan antara menghindari kerugian akibat False Positives dan risiko kehilangan pelanggan akibat False Negatives sangat penting. F1-Score membantu memastikan bahwa model cukup sensitif (precision) untuk mendeteksi pelanggan berisiko, namun tetap selektif (recall) agar promo diberikan secara efisien.

### Benchmark Model: K-Fold

![Gambar14](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar14.jpg)

Setelah cross-validation, hasil evaluasi untuk masing-masing model (`Logistic Regression`, `KNN`, `Decision Tree`, `Random Forest`, `XGBoost`, dan `LightGBM`) ditampilkan. Hasilnya adalah DataFrame yang menampilkan F1-macro score untuk setiap model, yang digunakan untuk menentukan model terbaik berdasarkan kinerja mereka.

### Benchmark Model: Test Data

Hasilnya adalah tabel yang menunjukkan evaluasi dari berbagai model klasifikasi berdasarkan metrik, seperti **precision**, **recall**, **f1-score**, **accuracy**, dan **weighted avg** untuk setiap model yang diuji. Tabel yang ditampilkan mengurutkan model berdasarkan skor **F1-macro**, sehingga model dengan performa terbaik (berdasarkan F1-macro) akan berada di atas. **XGBoost** memiliki F1-macro tertinggi (0.960177), diikuti oleh **Random Forest**, **Decision Tree**, **LightGBM**, dan **Logistic Regression**.

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

Laporan klasifikasi yang ditampilkan memberikan informasi tentang precision, recall, F1-score, dan support untuk setiap kelas sebelum tuning hyperparameter, serta informasi serupa untuk model XGBoost yang telah di-tuning dengan hyperparameter terbaik yang ditemukan selama pencarian acak.

![Gambar21](https://github.com/gunawan-ganda/Proyek-Pertama/blob/main/Gambar21.jpg)

Model XGBoost yang telah disesuaikan menunjukkan performa luar biasa dengan **f1-score macro sebesar 0.97**, yang mencerminkan keseimbangan antara presisi dan recall di kedua kelas (churn dan tidak churn). Dengan tingkat akurasi keseluruhan 98%, model ini sangat andal dalam memprediksi pelanggan yang berpotensi churn (kelas 1) maupun yang tetap loyal (kelas 0). Namun, mempertahankan **f1-score** di tingkat ini sangat penting karena secara langsung memengaruhi kemampuan untuk menangani churn pelanggan secara akurat.

## Kesimpulan

1. Pelanggan yang berisiko churn seringkali adalah mereka yang merasa kurang terlibat secara emosional maupun fungsional dengan layanan e-commerce. Mereka cenderung mencari pengalaman yang lebih baik, harga yang lebih kompetitif, atau layanan yang lebih sesuai dengan kebutuhan spesifik mereka. Mereka selalu menjelajahi opsi lain karena merasa layanan saat ini tidak cukup memuaskan, baik dari segi kenyamanan, kecepatan, atau insentif yang mereka harapkan. Mereka cenderung memiliki ekspektasi tinggi, tetapi dengan toleransi rendah terhadap ketidaksesuaian, menjadikan mereka kelompok yang memerlukan perhatian khusus dan pendekatan personal agar tetap setia.
2. Model XGBoost yang telah disesuaikan menunjukkan performa luar biasa dengan **f1-score macro sebesar 0.97**, yang mencerminkan keseimbangan antara presisi dan recall di kedua kelas (churn dan tidak churn). Dengan tingkat akurasi keseluruhan 98%, model ini sangat andal dalam memprediksi pelanggan yang berpotensi churn (kelas 1) maupun yang tetap loyal (kelas 0). Namun, mempertahankan **f1-score** di tingkat ini sangat penting karena secara langsung memengaruhi kemampuan untuk menangani churn pelanggan secara akurat.
3. Penerapan sistem Machine Learning berbasis XGBoost telah memberikan peningkatan signifikan dalam efisiensi prediksi risiko churn, dengan F1-Score sebesar 0.94 pada yang berisiko churn, mencerminkan keseimbangan optimal antara precision (0.94) dan recall (0.94).

# Gold Price Prediction with Predictive Analytics

## Domain Proyek
Emas telah digunakan sebagai bentuk mata uang di berbagai belahan dunia. Saat ini, logam mulia seperti emas dipegang oleh bank sentral di semua negara untuk menjamin pembayaran utang luar negeri, dan juga untuk mengendalikan inflasi yang mencerminkan kekuatan keuangan negara.

Emas telah menjadi salah satu komoditas yang diprioritaskan dalam hal investasi jangka panjang maupun jangka pendek. Memprediksi kenaikan dan penurunan harga emas harian dapat membantu investor memutuskan kapan harus membeli (atau menjual) komoditas. Tetapi, harga Emas juga bergantung pada banyak faktor seperti harga logam mulia lainnya, harga minyak mentah, kinerja bursa saham, harga Obligasi, nilai tukar mata uang, dan lain lain  [(Shafiee S, 2010)](https://www.sciencedirect.com/science/article/abs/pii/S0301420710000243)

## Business Understanding
### Problem Statement
- Apa saja faktor/fitur yang memiliki korelasi dengan target kelas (Adjusted close)?
- Bagaimana memprediksi adjusted close dari emas berdasarkan berbagai faktor/fitur menggunakan pendekatan machine learning/deep learning?

### Goals
Tujuan dari proyek ini adalah membuat pendekatan predictive analytics dengan machine learning untuk memprediksi harga adjusted close emas

### Solution Statements
Solusi dari permasalahan adalah menggunakan regresi. Kegunaan analisis regresi adalah untuk mengetahui variabel-variabel kunci yang memiliki pengaruh terhadap suatu variabel bergantung, pemodelan, serta pendugaan (estimation) atau peramalan (forecasting). Terdapat beberapa algoritma regresi yang digunakan pada proyek ini antrana lain
- **Linear Regresi**, persamaan linier yang menggabungkan satu set tertentu dari nilai input (x) dan solusi yang merupakan output yang diprediksi untuk set nilai input (y). Misalnya, dalam masalah regresi sederhana (satu x dan satu y), bentuk modelnya adalah:
y = B0 + B1*x.  Dengan demikian, baik nilai input (x) dan nilai output adalah numerik
- **Random Forest**, merupakan model ensemble bagging, random forest pada dasarnya adalah versi bagging dari algoritma decision tree. Teknik pembagian data pada algoritma decision tree adalah memilih sejumlah fitur (misal x kolom) dan sejumlah sampel (misal y baris) secara acak dari dataset yang terdiri dari (misalnya) n fitur dan m contoh. pada kasus regresi, prediksi akhir adalah rata-rata prediksi seluruh pohon dalam model ensemble.
- **Gradient Boosting**, merupakan model ensemble boosting, algoritma boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.

## Data Understanding
Dataset diambil dari [Kaggle Gold Price Prediction Dataset](https://www.kaggle.com/sid321axn/gold-price-prediction-dataset). Dataset tersebut dikumpulkan dari 18 November 2011 hingga 1 Januari 2019 dengan total 1718 baris dan 80 kolom. Terdapat data faktor tambahan seperti harga minyak, Standard and Poor's (S&P) 500 index, Dow Jones Index US Bond rates (10 tahun), Euro USD exchange rates, harga logam mulia seperti Silver dan Platinum, dan logam lain seperti Palladium and Rhodium
Fitur-fitur dari dataset antara lain:
- Date: tanggal harian trading
- Open: harga pembukaan harian
- High: harga tertinggi harian
- Low: harga terendah harian
- Close: harga penutupan harian
- Volume: ukuran total saham yang berpindah tangan untuk jangka waktu tertentu
- Gold ETF :- Date, Open, High, Low, Close and Volume.
- S&P 500 Index :- 'SP_open', 'SP_high', 'SP_low', 'SP_close', 'SP_Ajclose', 'SP_volume'
- Dow Jones Index :- 'DJ_open','DJ_high', 'DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume'
- Eldorado Gold Corporation (EGO) :- 'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume'
- EURO - USD Exchange Rate :- 'EU_Price','EU_open', 'EU_high', 'EU_low', 'EU_Trend'
- Brent Crude Oil Futures :- 'OF_Price', 'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend'
- Crude Oil WTI USD :- 'OS_Price', 'OS_Open', 'OS_High', 'OS_Low', 'OS_Trend'
- Silver Futures :- 'SF_Price', 'SF_Open', 'SF_High', 'SF_Low', 'SF_Volume', 'SF_Trend'
- US Bond Rate (10 years) :- 'USB_Price', 'USB_Open', 'USB_High','USB_Low', 'USB_Trend'
- Platinum Price :- 'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low','PLT_Trend'
- Palladium Price :- 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low','PLD_Trend'
- Rhodium Prices :- 'RHO_PRICE'
- US Dollar Index : 'USDI_Price', 'USDI_Open', 'USDI_High','USDI_Low', 'USDI_Volume', 'USDI_Trend'
- Gold Miners ETF :- 'GDX_Open', 'GDX_High', 'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume'
- Oil ETF USO :- 'USO_Open','USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume'

## Data Preparation
Data preparation adalah tahap di mana kita melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahapan yang dilakukan pada data preparation, antara lain:
* Seleksi fitur, menganalisa fitur apa saja yang mempengaruhi hasil target dengan mutual_info_regression
* train test split, membagi data menjadi data training dan testing dengan rasio 80%:20%
* standarisasi, membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma, pada proyek ini menggunakan algoritma z-score.
* dimensionality reduction dengan PCA, mengubah fitur asli menjadi kumpulan fitur lain yang tidak berkorelasi linier
Dari hasil data preparation, terdapat beberapa fitur yang redundan / tidak berguna seperti Volume, dan Trend. kemudian diambil 10 fitur yang paling mempengaruhi untuk dijadikan data yaitu, High, Low, Open, GDX_High, GDX_Close, GDX_Low, GDX_Adj Close, GDX_Open, SF_Price, dan SP_AjClose

## Modelling
Pada tahap ini digunakan 3 model regresi yaitu linear regression (scikit learn), random forest (scikit learn), dan gradient booster (xgboost), kemudian performa dari tiap model dibandingkan berdasarkan metrics hasil prediksi. Dari hasil pemodelan, dapat disimpulkan bahwa model linear regression mendapatkan error rate paling rendah berdasarkan MSE dan MAE nya.

## Evaluation
Kerakteristik utama dari masalah regresi adalah target dari kumpulan data hanya berisi bilangan real. Error menggambarkan seberapa banyak model membuat kesalahan dalam prediksinya, kemudian dibandingan dengan target aktual menurut metrik tertentu. Metrik evaluasi yang digunakan pada kasus regresi di proyek ini adalah
- Mean Absolute Error (MAE), merepresentasikan rata-rata perbedaan mutlak antara nilai aktual dan prediksi pada dataset. MAE mengukur rata-rata residu dalam dataset. MAE lebih intuitif dalam memberikan rata-rata error dari keseluruhan data.

![MAE](https://1.bp.blogspot.com/-OY4iwFkwEdQ/X8J8nmJFPFI/AAAAAAAACYo/hFjo4vbDdWguXH5XKhHEXWihbKKIkZA_wCLcBGAsYHQ/s241/Rumus%2BMAE.jpg)
- Mean Squared Error, merepresentasikan rata-rata perbedaan kuadrat antara nilai aktual dan prediksi pada dataset. MSE mengukur varians dari residual. MSE sangat baik dalam memberikan gambaran terhadap seberapa konsisten model yang dibangun karena model dengan varian kecil dapat memberi hasil yang relatif konsisten. namun MSE sangat sensitif dengan outlier.
untuk menerapkan metrik tersebut dapat menggunakan library Scikit Learn. kemudian membuat plot hasil prediksi dan aktual dari data testing untuk membandingan hasilnya secara visual.
![MSE](https://1.bp.blogspot.com/--Ktw4spozkk/X8J61DTY_2I/AAAAAAAACYc/syREhWmXAWA22_uhAo1e4DwBcRulroEjwCLcBGAsYHQ/s277/Rumus%2BMSE.jpg)









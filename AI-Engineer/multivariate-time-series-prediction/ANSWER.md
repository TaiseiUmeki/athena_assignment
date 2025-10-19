### プログラム概要

#### 各プログラムの概要と実行方法

- **ファイル EDA**: EDA に用いたプログラムをまとめたファイル

  - **basic_eda.py**:
    - **概要**: データの基本統計量を出力する。
    - **実行方法**:ターミナルで python ~/assignment-main/AI-Engineer/multivariate-time-series-prediction/eda/basic_eda.py を実行する。
  - **line.py**:
    - **概要**: 横軸を日時、縦軸を油温とした、データの折れ線図を出力する。
    - **実行方法**:ターミナルで python ~/assignment-main/AI-Engineer/multivariate-time-series-prediction/eda/line.py を実行する。
  - **acr.py**:
    - **概要**: データの自己相関関数の結果を出力する。
    - **実行方法**:ターミナルで python ~/assignment-main/AI-Engineer/multivariate-time-series-prediction/eda/acr.py を実行する。
  - **seasonal.py**:
    - **概要**: データの年季節成分について STL 分解を実行する。
    - **実行方法**:ターミナルで python ~/assignment-main/AI-Engineer/multivariate-time-series-prediction/eda/seasonal.py を実行する。

- **miss_inputation.py**:
  - **概要**: 観測データのうち、油温データ中の欠損値の数を出力するファイル。
  - **実行方法**:ターミナルで python ~/assignment-main/AI-Engineer/multivariate-time-series-prediction/miss_inputation.py を実行する。
- **sarimax.py**:
  - **概要**: SARIMAX について記述したファイル。
  - **実行方法**:ターミナルで python ~/assignment-main/AI-Engineer/multivariate-time-series-prediction/sarimax.py を実行する。
- **sarimax_nonfourier.py**:
  - **概要**: 仮説 1 フーリエ項なしの SARIMAX について記述したファイル。
  - **実行方法**:ターミナルで python ~/assignment-main/AI-Engineer/multivariate-time-series-prediction/sarimax_nonfourier.py を実行する。
- **arimax.py**:
  - **概要**: 仮説 2 季節性を全てフーリエ項で表現した　 ARIMAX について記述したファイル。
  - **実行方法**:ターミナルで python ~/assignment-main/AI-Engineer/multivariate-time-series-prediction/arimax.py を実行する。

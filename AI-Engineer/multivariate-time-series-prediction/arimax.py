import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# 予測・評価の長さ（ここだけ変更）
H = 720  # 4ヶ月 ≒ 120日×24h

# データ読み込み
df = pd.read_csv('/Users/umeking5/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv')
df = df.iloc[:, [0, 7]]  # date と OT
df['date'] = pd.to_datetime(df['date'])

df_test  = df.iloc[-H:]
df_train = df.iloc[:-H]

# 説明変数X（Fourier terms）の生成
exog = pd.DataFrame()
exog.index = df['OT'].index

def fourier_terms_gen(seasonal, terms_num):
    for num in range(1, terms_num + 1):
        exog[f'sin{seasonal}_{num}'] = np.sin(num * 2 * np.pi * exog.index / seasonal)
        exog[f'cos{seasonal}_{num}'] = np.cos(num * 2 * np.pi * exog.index / seasonal)

# 24周期
fourier_terms_gen(seasonal=24, terms_num=10)
# 8760周期
fourier_terms_gen(seasonal=8760, terms_num=10)

exog_train = exog.iloc[:-H]
exog_test  = exog.iloc[-H:]

# モデル構築
arima_fourier_model = SARIMAX(df_train['OT'], order=(2,1,2), exog=exog_train)
arimax_result = arima_fourier_model.fit()

# 予測
forecast = arimax_result.get_forecast(steps=H, exog=exog_test)
predicted_mean = forecast.predicted_mean

# 評価
rmse = np.sqrt(np.mean((df_test['OT'].values - predicted_mean.values) ** 2))
print('RMSE:', rmse)
mae = np.mean(np.abs(df_test['OT'].values - predicted_mean.values))
print('MAE:', mae)
mape = np.mean(np.abs((df_test['OT'].values - predicted_mean.values) / df_test['OT'].values)) * 100
print('MAPE(%):', mape)

# グラフ化（テストデータの期間）
fig, ax = plt.subplots()
ax.plot(df_test.index, df_test['OT'].values, label="actual(test dataset)")
ax.plot(df_test.index, predicted_mean.values, label="SARIMA", linestyle="dotted", lw=2, color="m")
plt.legend()
plt.show()

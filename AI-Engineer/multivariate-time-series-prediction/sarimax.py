import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# 予測・評価の長さ
H = 720  # 1ヶ月 ≒ 120日×24h

df = pd.read_csv('/Users/umeking5/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv')
df = df.iloc[:, [0,7]]  # dateとOT列のみ抽出
df['date'] = pd.to_datetime(df['date'])

# 学習/評価の分割を H に合わせて変更
df_test  = df.iloc[-H:]
df_train = df.iloc[:-H]

# 説明変数Xの生成
seasonal_year = 8760   # 年

exog = pd.DataFrame()
exog.index = df['OT'].index

t = np.arange(len(exog))

# 年周期のFourier terms
exog['sinyear_1'] = np.sin(1 * 2 * np.pi * t / seasonal_year)
exog['cosyear_1'] = np.cos(1 * 2 * np.pi * t / seasonal_year)
exog['sinyear_2'] = np.sin(2 * 2 * np.pi * t / seasonal_year)
exog['cosyear_2'] = np.cos(2 * 2 * np.pi * t / seasonal_year)

# exog も H に合わせてスライス
exog_train = exog.iloc[:-H]
exog_test  = exog.iloc[-H:]

sarimax_model = SARIMAX(df_train['OT'], order=(2,1,2), seasonal_order=(1,1,1,24), exog=exog_train)
sarimax_result = sarimax_model.fit(disp=False)

# 予測（H ステップ先）
forecast = sarimax_result.get_forecast(steps=H, exog=exog_test)
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

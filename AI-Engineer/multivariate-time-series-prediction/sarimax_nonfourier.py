import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# 予測・評価の長さ
H = 720  # 1ヶ月 ≒ 120日×24h

# データ読み込み
df = pd.read_csv('/Users/umeking5/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv')
df = df.iloc[:, [0, 7]]  # date と OT
df['date'] = pd.to_datetime(df['date'])

# 学習/評価データに分割（直近H時間をテスト）
df_train = df.iloc[:-H]
df_test  = df.iloc[-H:]

# モデル（フーリエ項なし）
model = SARIMAX(
    df_train['OT'],
    order=(2, 1, 2),
    seasonal_order=(0, 1, 1, 24)  # 日周期のみ
)
res = model.fit(disp=False)

# 予測
fcst = res.get_forecast(steps=H)
pred = fcst.predicted_mean

# 評価
rmse = np.sqrt(np.mean((df_test['OT'].values - pred.values) ** 2))
mae  = np.mean(np.abs(df_test['OT'].values - pred.values))
mape = np.mean(np.abs((df_test['OT'].values - pred.values) / df_test['OT'].values)) * 100

print('order:', model.order)
print('seasonal_order:', model.seasonal_order)
print('RMSE:', rmse)
print('MAE:', mae)
print('MAPE(%):', mape)

# グラフ化（テストデータの期間）
fig, ax = plt.subplots()
ax.plot(df_test.index, df_test['OT'].values, label="actual(test dataset)")
ax.plot(df_test.index, pred.values, label="SARIMA", linestyle="dotted", lw=2, color="m")
plt.legend()
plt.show()

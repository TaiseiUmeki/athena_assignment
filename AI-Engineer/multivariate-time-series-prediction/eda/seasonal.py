import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.dates as mdates

# CSV読み込み
df = pd.read_csv('/Users/umeking5/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv')

# 日付をdatetime型に変換・インデックス設定
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# STL分解（1年 = 8760時間周期）
stl = STL(df['OT'], period=8760)
stl_series = stl.fit()

# 結果をDataFrame化
stl_df = pd.DataFrame({
    'observed': stl_series.observed,
    'trend': stl_series.trend,
    'seasonal': stl_series.seasonal,
    'resid': stl_series.resid
}, index=df.index)

# プロット
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
titles = ['Observed', 'Trend', 'Seasonal', 'Residual']

for ax, col, title in zip(axes, stl_df.columns, titles):
    ax.plot(stl_df.index, stl_df[col], label=title)
    ax.set_title(title)
    ax.legend()

# ← 横軸を「年月（YYYY-MM）」で表示する設定
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 2か月ごとにラベル表示
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 季節性の強さを確認
resid_std = stl_df['resid'].std() # 残差の標準偏差
seasonal_std = stl_df['seasonal'].std() # 季節性成分の標準偏差
seasonal_strength = seasonal_std / resid_std # 季節性の強さ
print(f"季節性の強さ: {seasonal_strength:.2f}")

# データ全体に対する季節性の寄与度
total_variance = stl_df['observed'].var()  # 全体の分散
seasonal_variance = stl_df['seasonal'].var()  # 季節性成分の分散
seasonal_ratio = seasonal_variance / total_variance  # 季節性分散の割合
print(f"季節性の寄与度: {seasonal_ratio:.2%}")
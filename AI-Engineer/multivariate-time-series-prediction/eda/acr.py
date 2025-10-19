import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# CSVファイルを読み込む
df = pd.read_csv('/Users/umeking5/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv')
date = pd.to_datetime(df['date'])
ot = df['OT']

# トレンド非定常性を除去
ot_diff = ot.diff().dropna()

#差分をとった時系列データのプロット
#plt.figure(figsize=(12, 6))
#plt.plot(ot_diff, label='Differenced OT', color='blue')
#plt.title('Differenced OT over Date')
#plt.xlabel('Date')
#plt.ylabel('Differenced OT')
#plt.show()

# グラフを表示する領域と２つのサブプロットのセットを作成
fig, axes = plt.subplots(2, 1, figsize=(12, 12))


# ACFを計算し、1つ目のサブプロットにプロット
plot_acf(ot_diff, lags=200, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)  of Differenced Data')  # Set the title
axes[0].grid(True)  # Display the grid lines

# PACFを計算し、2つ目のサブプロットにプロット
plot_pacf(ot_diff, lags=200, method='ywm', ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)  of Differenced Data')  # Set the title
axes[1].grid(True)  # Display the grid lines

# プロットを表示
plt.tight_layout()
plt.show()
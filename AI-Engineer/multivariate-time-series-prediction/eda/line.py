import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('/Users/umeking5/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv')
date = pd.to_datetime(df['date'])

plt.plot_date(date, df['OT'], fmt='-r', tz=None, xdate=True, ydate=False)
plt.title('Plot of OT over Date')
plt.xlabel('date')
plt.ylabel('OT')
#plt.xticks(rotation=45)
#plt.tight_layout()
plt.show()
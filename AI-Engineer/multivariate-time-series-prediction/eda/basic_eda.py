import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.display.max_columns = 32

# CSVファイルを読み込む
Ett = pd.read_csv('/Users/umeking5/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv')

print(Ett.head(8))
#Ett.tail()
#Ett.describe()

#基本統計量を表示
df = pd.read_csv('/Users/umeking5/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv')
#print(df.describe())


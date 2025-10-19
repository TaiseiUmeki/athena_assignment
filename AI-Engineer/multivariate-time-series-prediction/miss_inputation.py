import pandas as pd

df = pd.read_csv('/Users/umeking5/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv')
df['date'] = pd.to_datetime(df['date'])
ot = df['OT']
#print(df.isnull().sum())
print(df.info())
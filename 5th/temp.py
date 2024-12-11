
import pandas as pd

df=pd.read_csv("encoded_departs.csv", nrows=1000)
df = df.drop(df.iloc[:, 300:],axis = 1)

print(df.columns.values)
import pandas as pd
print('a')
a=pd.read_csv(r"C:\\Users\\winlabuser\\Downloads\\usa_00001.dat\\myauto.csv")#, nrows=999)
print(a.head)
print(a.iloc(0))

print(a.columns)
print(a.shape)
print(a.dtypes)
b=pd.get_dummies(a)
print(b.head)
print(b.columns)
print(b.shape)
print(b.dtypes)



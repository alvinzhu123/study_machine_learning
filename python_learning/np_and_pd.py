import numpy as np
import pandas as pd
# #concate
# a = np.array([1,1,1])
# a = a[:,np.newaxis]
# b = np.concatenate((a,a,a), axis=0)
# print(b)
# #copy & deep copy
# a = np.arange(4)
# b = a
# c = b
# d = a[:]
# e = np.copy(a) #deep copy
# a[0] = 11
# print(b)
# print(c)
# print(d)
# print(e)

dates = pd.date_range("2020-8-01", periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A', "B", 'C', 'D'])
print(df)
# #select by label
# print(df.loc['2020-08-01'])
# #select by position
# print(df.iloc[[1,3,5], 1:3])
df.iloc[1,3] = 111
df.B[df.A>4] = 0
print(df)

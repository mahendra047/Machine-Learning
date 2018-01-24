# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:24:53 2017

@author: Mahendra Thakur
"""

import pandas as pd
pd.Series()
import numpy as np
a=np.array(['a','xyx'])
a
print(a)
ser=pd.Series(a,index=[50,75])
ser
import pandas as pd
import numpy as np
data = np.array(['a','b','c','d'])
s = pd.Series(data)
print (s)
#import the pandas library and aliasing as pd
import pandas as pd
import numpy as np
data = {'a' : 0., 'b' : 1., 'c' : 2.}
s = pd.Series(data)
print (s)
data = {'a' : 0., 'b' : 1., 'c' : 2.}
s = pd.Series(data,index=['b','c','d','a'])
print( s)
s = pd.Series(5, index=[0, 1, 2, 3])
print (s)
s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])
s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])
print( s[['a','c','d']])
df = pd.DataFrame()
print( df)
data = [1,2,3,4,5]
df = pd.DataFrame(data)
print (df)
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'])
print (df)

data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
print (df)

data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
print (df)

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print( df)

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df1 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b'])
df2 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b1'])
print( df1)
print (df2)

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print (df)


print(df['one'])

data = np.random.rand(2,4,5)
p = pd.Panel(data)
print (p)

import pandas as pd
import numpy as np
data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)), 
        'Item2' : pd.DataFrame(np.random.randn(4, 2))}
p = pd.Panel(data)
print (p['Item1'])

s = pd.Series(np.random.randn(4))
s = pd.Series(np.random.randn(4))
print ("The axes are:")
print (s.axes)

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

#Create a DataFrame
df = pd.DataFrame(d)
print ("Our data series is:")
print (df)
print (df.T)
print ("The dimension of the object is:")
print( df.shape)

print( df.values)

print( df.size)


print( df.ndim)

print (df.sum())
print (df.sum(1))
print (df.mean())
print( df.describe())
print (df. describe(include='all'))
 
def adder(ele1,ele2):
    return (ele1+ele2)

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df
df.pipe(adder,2)

import pandas as pd
import numpy as np

def adder(ele1,ele2):
   return ele1+ele2

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.pipe(adder,2)
print (df)

df.apply(np.mean)
print (df)

df.apply(np.mean,axis=1)
print( df)

df.apply(lambda x: x.max() - x.min())
print (df)

df['col1'].map(lambda x:x*100)
print(df)

df['col2'].map(lambda x:x/10)

df.applymap(lambda x:x*100)
df.apply(lambda x:x*100)

N=20

df = pd.DataFrame({
   'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
   'x': np.linspace(0,stop=N-1,num=N),
   'y': np.random.rand(N),
   'C': np.random.choice(['Low','Medium','High'],N).tolist(),
   'D': np.random.normal(100, 10, size=(N)).tolist()
})

print(df)
df_reindexed = df.reindex(index=[0,2,5], columns=['A', 'C', 'B'])

print (df_reindexed)


df1 = pd.DataFrame(np.random.randn(10,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(7,3),columns=['col1','col2','col3'])
print(df1)
print(df2)
df1 = df1.reindex_like(df2)
print (df1)

import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(2,3),columns=['col1','col2','col3'])

# Padding NAN's
print (df2.reindex_like(df1))

# Now Fill the NAN's with preceding Values
print ("Data Frame with Forward Fill:")
print (df2.reindex_like(df1,method='ffill'))

df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(2,3),columns=['col1','col2','col3'])

# Padding NAN's
print (df2.reindex_like(df1))

# Now Fill the NAN's with preceding Values
print ("Data Frame with Forward Fill limiting to 1:")
print(df2.reindex_like(df1,method='ffill',limit=1))

df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
print (df1)

print ("After renaming the rows and columns:")
print (df1.rename(columns={'col1' : 'c1', 'col2' : 'c2'},
index = {0 : 'apple', 1 : 'banana', 2 : 'durian'}))


N=20

df = pd.DataFrame({
    'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
    'x': np.linspace(0,stop=N-1,num=N),
    'y': np.random.rand(N),
    'C': np.random.choice(['Low','Medium','High'],N).tolist(),
    'D': np.random.normal(100, 10, size=(N)).tolist()
    })

for col in df:
   print (df)

df = pd.DataFrame(np.random.randn(4,3),columns=['col1','col2','col3'])
for key,value in df.iteritems():
   print( key,value)




























































































































































































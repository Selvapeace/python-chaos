import pandas as pd
import numpy as np

# 创建数据框
s = pd.Series([1,3,4,5,6,7,np.nan,44,3])
print(s)

dates = pd.date_range('20180101',periods = 6)
print(dates)
df0 = pd.DataFrame(np.random.randn(6,4),index = dates,columns = ['a','b','c','d'])
print(df0)
df1 = pd.DataFrame(np.arange(12).reshape(3,4))     # 默认用0开始的数字来做行、列标签
print(df1)
df2 = pd.DataFrame({'a':1.,
                    'b':pd.Timestamp('20180101'),
                    'c':pd.Series(1,index = list(range(4)),dtype = 'float32'),
                    'd':np.array([3]*4,dtype = 'int32'),
                    'e':pd.Categorical(['test','train','test','train']),
                    'f':'foo'})
print(df2)
print(df2.dtypes)
print(df2.index)      # 行名
print(df2.columns)    # 列名
print(df2.values)     # 值
print(df2.describe())   # 数值型数据的描述性统计
print(df0.sort_index(axis = 1,ascending = False))    # 对列标签排序
print(df0.sort_index(axis = 0,ascending = False))    # 对行标签排序
print(df2.sort_values(by = 'e'))       # 对值排序

print('\n','\n','\n','\n')


# 从数据框中选取数据
df = pd.DataFrame(np.arange(24).reshape((6,4)),index = dates,columns = ['a','b','c','d'])
print(df)
## 通过标签来选取
print(df['a'],'\n',df.a)
print(df[0:3],'\n',df['20180102':'20180104'])
print(df.loc['20180101'])
print(df.loc[:,['a','b']])
## 通过位置来选取
print(df.iloc[3])
print(df.iloc[3:5,2])
print(df.iloc[[1,2,5],2])
## 混合选取
print(df.ix[:3,['a','c']])
## 条件选取
print(df[df.a>8])


# 设置值
df.iloc[2,2] = 1111
print(df)
df.loc['20180101','b'] = 222
print(df)
df.a[df.a>0] = 1
print(df)
df.c[df.a>0] = 3
print(df)
df['f'] = np.nan     # 增加列
df['e'] = pd.Series([1,3,4,np.nan,44,3],index = pd.date_range('20180101',periods = 6))
print(df)

# 处理nan
df = pd.DataFrame(np.arange(24).reshape((6,4)),index = dates,columns = ['a','b','c','d'])
df['e'] = pd.Series([1,3,4,np.nan,44,3],index = pd.date_range('20180101',periods = 6))
print(df.dropna(axis = 0,how = 'any'))   # 把带有nan的行丢掉
print(df.dropna(axis = 1,how = 'all'))   # 把所有数据都是nan的列丢掉
print(df.fillna(value = 0))     # 把nan替换成0
print(df.isnull())     # 判断是否有nan
print(np.any(df.isnull() == True))    # np.any() 是否至少有一个True


'''
# 读取数据
data = pd.read_csv('student.csv')
# 存储数据
data.to_excel('student1.xlsx')
'''

# 合并数据框concatenation
## example 1
df1 = pd.DataFrame(np.ones((3,4))*0,columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns = ['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2,columns = ['a','b','c','d'])
print(df1,'\n',df2,'\n',df3)
res = pd.concat([df1,df2,df3],axis = 0,ignore_index = True)   # 纵向合并，把行标签重新命名
print(res)
## example 2
df1 = pd.DataFrame(np.ones((3,4))*0,columns = ['a','b','c','d'],index = [1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1,columns = ['a','b','e','d'],index = [6,2,8])
print(df1,'\n',df2,'\n')
res = pd.concat([df1,df2],join = 'outer')     # 这样合并后会把相互没有的元素变为nan；默认outer
print(res)
res = pd.concat([df1,df2],join = 'inner',ignore_index = True)     # 相互没有的部分直接删除
print(res)
res = pd.concat([df1,df2],axis = 1,join_axes = [df1.index])    # 横向合并，行标签用df1的行标签，没有对应的元素用nan填充
print(res)
res = pd.concat([df1,df2],axis = 1)    # 横向合并，行标签用df1、df2的行标签，没有对应的元素用nan填充
print(res)
## append
df1 = pd.DataFrame(np.ones((3,4))*0,columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns = ['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2,columns = ['a','b','c','d'])
res = df1.append([df2,df3],ignore_index = True)
print(res)
s1 = pd.Series([1,2,3,4],index = ['a','b','c','d'])
res = df1.append(s1,ignore_index = True)
print(res)

# 合并数据框merge
## example 1
left = pd.DataFrame({'key':['K0','K1','K2','K3'],
                     'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','B3']})
right = pd.DataFrame({'key':['K0','K1','K2','K3'],
                     'C':['C0','C1','C2','C3'],
                     'D':['D0','D1','D2','D3']})
print(left,'\n',right)
res = pd.merge(left,right,on = 'key')    # 根据key标签合并
print(res)
## example 2
left = pd.DataFrame({'key1':['K0','K0','K1','K2'],
                     'key2':['K0','K1','K0','K1'],
                     'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','B3']})
right = pd.DataFrame({'key1':['K0','K1','K1','K2'],
                     'key2':['K0','K0','K0','K0'],
                     'C':['C0','C1','C2','C3'],
                     'D':['D0','D1','D2','D3']})
print(left,'\n',right)
res = pd.merge(left,right,on = ['key1','key2'],how = 'inner')   # inner是默认
print(res)
res = pd.merge(left,right,on = ['key1','key2'],how = 'outer')
print(res)
res = pd.merge(left,right,on = ['key1','key2'],how = 'left')    # 根据left的key来填充
print(res)
## example 3 indicator
df1 = pd.DataFrame({'col1':[0,1],'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
print(df1,'\n',df2)
res = pd.merge(df1,df2,on = 'col1',how = 'outer',indicator = True)
print(res)      # indicator 可以显示merge的情况
res = pd.merge(df1,df2,on = 'col1',how = 'outer',indicator = 'indicator_column')
print(res)
## example 4 按行标签合并
left = pd.DataFrame({'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','B3']},
                    index = ['K0','K1','K2','K3'])
right = pd.DataFrame({'C':['C0','C1','C2','C3'],
                     'D':['D0','D1','D2','D3']},
                     index = ['K0','K1','K2','K3'])
print(left,'\n',right)
res = pd.merge(left,right,left_index = True,right_index = True,how = 'outer')
print(res)    # left_index和right_index的默认值是False
## example 5 处理overlapping
boys = pd.DataFrame({'k':['K0','K1','K2'],'age':[1,2,3]})
girls = pd.DataFrame({'k':['K0','K0','K3'],'age':[4,5,6]})
print(boys,'\n',girls)
res = pd.merge(boys,girls,on = 'k',suffixes = ['_boys','_girls'],how = 'outer')
print(res)     # 对相同的标签增加副标签


# 绘图
import matplotlib.pyplot as plt
## series 绘图
data = pd.Series(np.random.randn(1000),index = np.arange(1000))
data = data.cumsum()
data.plot()
plt.show()
## dataframe 绘图
data = pd.DataFrame(np.random.randn(1000,5),index = np.arange(1000),
                    columns = list('ABCDE'))
data = data.cumsum()
print(data.head())
data.plot()
plt.show()

ax = data.plot.scatter(x = 'A',y = 'B',color = 'DarkBlue',label = 'class 1')
data.plot.scatter(x = 'A',y = 'C',color = 'DarkGreen',label = 'class 2',ax = ax)
plt.show()

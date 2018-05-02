import numpy as np

a = np.array([[1,2,3],[2,3,4]],dtype = np.int64)    # dtype = np.float32 等都可
print(a.dtype)
print('number of dim',a.ndim)
print('shape',a.shape)
print('size',a.size)

b = np.zeros((3,4))    # 创建3行4列的0矩阵
print(b)

c = np.ones((3,4))
print(c)

d = np.arange(10,20,2)    # 从10开始以2为步长直到20（20取不到），之间所有的数组成数列
print(d)

e = np.arange(12).reshape((3,4))
print(e)

f = np.linspace(1,10,6)     # 1到10平均分成6段
f = f.reshape((2,3))
print(f)

print(e-c)
print(e**2)    # 向量中的所有元素自身平方
print(10*np.sin(a))    # sin(a)对a所有元素自身求sin
print(e<3)      # 返回True和False组成的矩阵

print(np.dot(a,e))    # 对于用array定义的向量，np.dot表示矩阵乘法
print(a.dot(e))     # 同上
print(a*f)        # 对于用array定义的向量，*表示对应元素分别相乘

g = np.random.random((2,4))    # 随机矩阵
print(g)
print('...',np.sum(g),'\n',np.sum(g,axis = 1),'\n',np.max(g),'\n',np.max(g,axis = 0),'\n',np.min(g))
                        # 几乎所有方法都可以选择axis：axis = 1 表示对每行执行；axis = 0 表示对每列执行

print(np.argmin(e),'\n',np.argmax(e))      # 最大最小值的索引
print(np.mean(e),'\n',e.mean())
print(np.median(e))
print(np.cumsum(e))      # 逐个累加，每加一次，得到一个元素
print(np.diff(e))        # 逐个相减
print(np.sort(e))      # 从小到大排序
print(np.transpose(e),'\n',e.T)  # 矩阵转置
print(np.clip(e,5,9))     # 所有小于5的数变成5，所有大于9的数变成9


print(e[2,3],'\n',e[2][3])    # 3,4位置的元素
print(e[2],'\n',e[2:,])     # 第三行
print(e[:,3])        # 第四列
for row in e:
    print(row)     # 打印每一行
for column in e.T:
    print(column)     # 打印每一列
print(e.flatten())     # 把类型变成一维向量
for item in e.flat:
    print(item)


print(np.vstack((c,e)),'\n',np.vstack((c,e)).shape)     # vertical stack  上下合并
print(np.hstack((c,e)),'\n',np.hstack((c,e)).shape)     # horizontal stack 左右合并
print(np.concatenate((c,e,e,e),axis = 0))           # 合并，方向用axis控制
print(np.transpose([1,2,3]))    # n*1的向量用transpose转置无效
print(np.array([1,2,3]).shape)    # n*1向量的维度是(n,)
print((np.array([1,2,3]))[:,np.newaxis],(np.array([1,2,3]))[:,np.newaxis].shape)
                            # 实现n*1向量转置：在列上增加一个维度，变成(n,1)


print(np.split(e,2,axis = 1))    # 这里axis = 1 表示分成元素个数相等的横向的两块
print(np.array_split(e,3,axis = 1))     # 进行不等分割
print(np.vsplit(e,3))
print(np.hsplit(e,2))     # vsplit和hsplit都是相等分割


h = np.arange(4)    # 这样定义的向量默认为整数型，如果修改元素值如h[0] = 0.2，则该元素变为0
print(h)
hh = h
hhh = h
hhhh = hh
h[0] = 11
print(h,hh,hhh,hhhh)     # h的改变同时导致hh、hhh、hhhh改变
hhhh[1:3] = [22,33]
print(h,hh,hhh,hhhh)     # hhhh的改变同时导致hh、hhh、h改变——上述赋值导致四个变量完全等价
hh = h.copy()      # deep copy ，不会把h和hh关联起来
h[0] = 44
print(h,hh)       # hh没有因为h的改变而改变






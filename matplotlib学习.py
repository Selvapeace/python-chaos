import matplotlib.pyplot as plt
import numpy as np

# 线图
x = np.linspace(-3,3,50)    # -1到1等分的50个数
y1 = 2 * x + 1
y2 = x**2

plt.figure(num = 3,figsize = (8,5))    # plt.figure()之后所有的都绘制在同一张图中
plt.plot(x,y2,label = 'y=x^2')
plt.plot(x,y1,color = 'red',linewidth = 1.0,linestyle = '--',label = 'y=2*x+1')

plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am x')
plt.ylabel('I am y')
new_ticks = np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,0,1.22,3],
            [r'$really\ bad$',r'$bad$',r'$normal$',r'$\alpha$',r'$good$',r'$really\ good$'])
                    # 两侧用$包起来，空格用'\ '表示，可以把字体换成数学形式的字体！

ax = plt.gca()      # gca = get current axis
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

plt.legend(labels = ['aaa','bbb'],loc = 'best')   # loc可设置的值：'upper right','lower right'等

x0 = 0.5
y0 = 2 * x0 + 1
plt.scatter(x0,y0,color = 'b')

plt.show()


# 散点图
n = 1024
x = np.random.normal(0,1,n)     # 正态随机数
y = np.random.normal(0,1,n)
t = np.arctan2(y,x)     # 设置颜色（不同的值对应不同颜色）

plt.scatter(x,y,s = 75,c = t,alpha = 0.5)    # s是size,c是color,alpha是透明度
plt.xlim((-2.5,2.5))
plt.ylim((-2.5,2.5))
plt.xticks()     # 括号里空着表示隐藏坐标轴上所有点
plt.yticks()

plt.show()


# 条形图
n = 12
x = np.arange(n)
y1 = (1 - x / float(n)) * np.random.uniform(0.5,1.2,n)
y2 = (1 - x / float(n)) * np.random.uniform(0.5,1.2,n)

plt.bar(x,+y1,facecolor = '#9999ff',edgecolor = 'white')
plt.bar(x,-y2,facecolor = '#ff9999',edgecolor = 'white')

plt.xlim(-0.5,n)
plt.xticks()
plt.ylim(-1.25,1.25)
plt.yticks()

plt.show()


# 等高线图
def f(x,y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2,-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)

X,Y = np.meshgrid(x,y)    # 把x,y网格化放到地面
C = plt.contour(X,Y,f(X,Y),10,alpha = 0.75,cmap = plt.cm.cool)
            # cmap=colormap，是不同的颜色风格(hot,cool等);10表示等高线把整个图分成10+2=12个区域
plt.clabel(C,inline = True,fontsize = 10)

plt.xticks()
plt.yticks()
plt.show()


# 用图片表示数字
a = np.array([0.6486653846,0.2562328498,0.31561358,0.5318948,0.56,0.135]).reshape(2,3)

plt.imshow(a,interpolation = 'nearest',cmap = plt.cm.hot,origin = 'upper')
plt.colorbar(shrink = 0.9)

plt.xticks()
plt.yticks()
plt.show()


# 3D图像
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

X = np.arange(-4,4,0.25)
Y = np.arange(-4,4,0.25)
X,Y = np.meshgrid(X,Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X,Y,Z,rstride = 1,cstride = 1,cmap = plt.get_cmap('rainbow'))
ax.contourf(X,Y,Z,zdir = 'z',offset = -2,cmap = 'rainbow')   # 等高线从z轴方向映射，等高线放在z=-2处
ax.set_zlim(-2,2)

plt.show()


# 多合一显示(1)
plt.figure()

plt.subplot(2,2,1)   # 把整个画布分成2行2列，目前这个图放在第1个位置
plt.plot([0,1],[0,1])

plt.subplot(2,2,2)
plt.plot([0,1],[0,2])

plt.subplot(2,2,3)
plt.plot([0,1],[0,3])

plt.subplot(2,2,4)
plt.plot([0,1],[0,4])

plt.show()


# 多合一显示(2)

## method 1
plt.figure()
ax1 = plt.subplot2grid((3,3),(0,0),colspan = 3,rowspan = 1)    # 划分成3,3；从0,0位置开始
ax1.plot([1,2],[1,2])
ax1.set_xlim()    # 之后设置都要加个set
ax2 = plt.subplot2grid((3,3),(1,0),colspan = 1,rowspan = 1)
ax2.plot([1,2],[1,2])
ax3 = plt.subplot2grid((3,3),(1,1),colspan = 1,rowspan = 1)
ax3.plot([1,2],[1,2])
ax4 = plt.subplot2grid((3,3),(1,2),colspan = 1,rowspan = 3)
ax4.plot([1,2],[1,2])
ax5 = plt.subplot2grid((3,3),(2,0),colspan = 2,rowspan = 1)
ax5.plot([1,2],[1,2])

plt.show()

## method 2
import matplotlib.gridspec as gridspec
plt.figure()
gs = gridspec.GridSpec(3,3)
ax1 = plt.subplot(gs[0,:])
ax2 = plt.subplot(gs[1,:2])
ax3 = plt.subplot(gs[1:,2])
ax4 = plt.subplot(gs[-1,0])
ax5 = plt.subplot(gs[-1,-2])

plt.tight_layout()
plt.show()


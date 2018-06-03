import random
import math
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib as mpl
random.seed(0)  
  
  
def rand(a, b):
    #随机数函数  
    return (b - a) * random.random() + a  
  
  
def make_matrix(m, n, fill=0.0):
    #矩阵生成函数  
    mat = []  
    for i in range(m):  
        mat.append([fill] * n)  
    return mat  
  
  
def sigmoid(x):
    #激活函数  
    return 1.0 / (1.0 + np.exp(-x))  
  
  
def sigmoid_derivative(x):
    #激活函数求导 
    m,n = x.shape
    out = np.mat(np.zeros((m,n)))
    for i in range(m):
        for j in range(n):
            out[i,j] = sigmoid(x[i,j])*(1-sigmoid(x[i,j]))
    return out 

def get_circle(r,num):
    #生成圆形线性不可分数据
    num = int(num/4)
    xs = r*np.random.rand(num,1)
    data = []
    for i in range(num):
        data.append([xs[i,0],math.sqrt(r**2-(xs[i,0]**2))])
        data.append([-xs[i,0],math.sqrt(r**2-xs[i,0]**2)])
        data.append([xs[i,0],-math.sqrt(r**2-xs[i,0]**2)])
        data.append([-xs[i,0],-math.sqrt(r**2-xs[i,0]**2)])
    data = np.array(data)
    return data

def get_data(num=4000,p_ratio=0.5,val_ratio=0.25):
    #获取训练和测试数据
    '''
    parameter：
        num：数据数量
        p_ratio：正样本占比
        val_ratio：测试数据占比
    return：
        x_train,y_train,x_test,y_test
    '''
    r1 = get_circle(5,int(num*p_ratio))
    r2 = get_circle(10,int(num*(1-p_ratio)))
    l1 = np.ones(int(num*p_ratio))
    l2 = np.zeros(int(num*(1-p_ratio)))
    data = np.concatenate((r1,r2),axis=0)
    lable = np.concatenate((l1,l2),axis=0)
    index = [i for i in range(num)]
    random.shuffle(index)
    data = data[index]
    lable = lable[index]
    cut = int(num*(1-val_ratio))
    x_train = data[:cut]
    y_train = lable[:cut]
    x_test = data[cut:]
    y_test = lable[cut:]
    return x_train,y_train,x_test,y_test

def paint(x,y,net,loss,iters):
    #画loss曲线以及分类结果
    plt.figure(facecolor='w')
    plt.subplot(311)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('loss while training')
    plt.plot(iters,loss)
    plt.subplot(313)
    plt.title('classify result')
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  
    mpl.rcParams['axes.unicode_minus'] = False  
    cm_light = mpl.colors.ListedColormap(['#FFA0A0', '#A0FFA0'])  
    cm_dark = mpl.colors.ListedColormap(['#FFAAAA', '#AAAAFF']) 
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1 # x1的范围
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1 # x2的范围
    t1 = np.linspace(x1_min, x1_max, 100)
    t2 = np.linspace(x2_min, x2_max, 100)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis=1) 
    y_hat = net.predict(x_show) # 预测
    y_h = []
    for i in range(y_hat.shape[0]):
        y_h.append(y_hat[i,0])
    y_hat = np.array(y_h)

    y_hat = y_hat.reshape(x1.shape)
     
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, edgecolors='k', cmap=cm_dark)
    plt.show()  

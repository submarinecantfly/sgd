import numpy as np
import random
from network import BPNeuralNetwork
from tools import get_data,paint
from config import config
cfg = config()#载入参数

if __name__ =='__main__':
    print('loading data')
    x_train,y_train,x_test,y_test = get_data(cfg.NUM_OF_DATA,cfg.P_RATIO,cfg.VAL_RATIO)#生成数据

    net = BPNeuralNetwork()
    print('building network')
    net.setup(cfg.INPUT_SHAPE,cfg.HIDDEN_SHAPE,cfg.OUTPUT_SHAPE)#定义网络

    print('training with SGD')
    losses,iters = net.SGD(cfg.MAX_ITER,cfg.LOSS_THRE,cfg.LR,x_train,y_train,x_test,y_test) #训练网络

    if cfg.PAINT:
        print('painting')
        paint(x_test,y_test,net,losses,iters)#绘制结果




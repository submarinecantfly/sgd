import numpy as np
from tools import *
class BPNeuralNetwork:#BP神经网络类  
    def __init__(self):#初始化  
        self.input_n = 0  
        self.hidden_n = 0  
        self.output_n = 0  
         
          
  
    def setup(self, ni, nh, no):  
        #初始化输入、隐层、输出元数  
        self.input_n = ni  
        self.hidden_n = nh  
        self.output_n = no  
        # 初始化权重矩阵  
        self.input_weights = np.mat(np.zeros((self.input_n, self.hidden_n)))  
        self.output_weights = np.mat(np.zeros((self.hidden_n, self.output_n)))  
        # 随机初始化权重  
        for i in range(self.input_n):  
            for h in range(self.hidden_n):  
                self.input_weights[i,h] = rand(-0.2, 0.2)  
        for h in range(self.hidden_n):  
            for o in range(self.output_n):  
                self.output_weights[h,o] = rand(-2.0, 2.0)  
        # 初始化偏置  
        self.input_correction = np.mat(np.zeros((1, self.hidden_n)))  
        self.output_correction =np.mat(np.zeros((1, self.output_n)))  

    def predict(self, inputs): 
        #预测函数 
        self.h_in = inputs*self.input_weights
        for i in range(inputs.shape[0]):
            self.h_in[i,:]+=self.input_correction#计算隐层输入

        self.h_out = sigmoid(self.h_in)#计算隐层输出

        self.p_in = self.h_out*self.output_weights
        for i in range(self.p_in.shape[0]):
            self.p_in[i,:] += self.output_correction#计算输出层输入

        self.p_out = sigmoid(self.p_in)#计算输出    
        return self.p_out  
          
  
    def back_propagate(self, case, label, alpha):  
        
        output_out = self.predict(case)  # 正向传播
        #反向传播
        self.delta_output = -np.multiply((label-output_out),sigmoid_derivative(self.p_in))#隐层与输出层的残差
        self.delta_hidden = np.multiply((self.delta_output*self.output_weights.T),sigmoid_derivative(self.h_in))#输入层与隐层的残差
        self.output_weights = self.output_weights - alpha*(self.h_out.T*self.delta_output)#更新输出权重
        self.output_correction = self.output_correction - np.sum(self.delta_output,axis=0)*(1.0/case.shape[0])#更新输出偏置
        self.input_weights = self.input_weights - alpha*(case.T*self.delta_hidden)#更新输入权重
        self.input_correction = self.input_correction - np.sum(self.delta_hidden,axis=0)*(1.0/case.shape[0])#更新输入偏置
        
        # 求当前误差  
        loss = 0.0  
        for o in range(label.shape[0]):  
            loss += 0.5 * (label[o] - self.p_out[o]) ** 2
        return loss  

    def SGD(self,max_iter,loss_thre,lr,x_train,y_train,x_test,y_test):
        #随机梯度下降训练网络
        losses = []
        iters = []
        flag = 0
        for i in range(max_iter):
            ind = np.random.randint(0,x_train.shape[0],1)#随机采样
            x = x_train[ind]
            y = y_train[ind]
            
            self.back_propagate(case=x,label = y,alpha =lr)#反向传播训练网络
            if i%500==0:#性能评估
                pre = self.predict(x_test)
                loss=0
                for j in range(pre.shape[0]):
                    loss+=(pre[j,0]-y_test[j])**2
                loss = loss/pre.shape[0]
                losses.append(loss)
                iters.append(i)
                

                print('after {} iters get loss {} on test data'.format(i,loss))#输出日志
                if loss<loss_thre:#满足阈值条件训练完成
                    print('training is done')
                    return losses,iters
                if loss>losses[-1]:#提前停止训练防止过拟合
                    flag+=1
                    if flag>2:
                        print('over fitting,training is done')
                        return losses,iters
                else:flag = 0


        print('training is done')
        return losses,iters

          
          
  
    
      
    
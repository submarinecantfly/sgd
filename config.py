class config:
    def __init__(self):
        self.NUM_OF_DATA = 4000 #数据数量
        self.P_RATIO = 0.5 #正样本占比
        self.VAL_RATIO = 0.25 #测试集占比
        self.MAX_ITER = 20000 #最大迭代数
        self.LOSS_THRE = 0.001 #loss阈值
        self.LR = 0.01 #学习率
        self.INPUT_SHAPE = 2 #样本特征维数
        self.HIDDEN_SHAPE = 50 #隐层神经元数量
        self.OUTPUT_SHAPE = 1 #输出维数
        self.PAINT = True #是否画图
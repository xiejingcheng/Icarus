import numpy as np

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self
    
    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.updateOne(param)

    def updateOne(self, param):
        raise NotImplementedError()
    
    def addHook(self, f):
        self.hooks.append(f)

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def updateOne(self, param):
        param.data -= self.lr * param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr 
        self.momentum = momentum
        self.vs = {}

    def updateOne(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        
        v = self.vs[v_key]
        v *= self.momentum 
        v -= self.lr * param.grad.data 
        param.data += v

class Adam(Optimizer):
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    https://www.jianshu.com/p/62f2df588cb1
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        super().__init__()
        self.lr = lr 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.iter = 0 
        self.ms = {}
        self.vs = {}
    
    def updateOne(self, param):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)
        
        m, v = self.ms[key], self.vs[key]
        beta1, beta2 = self.beta1, self.beta2 
        lr = self.lr 
        self.iter += 1 
        grad = param.grad.data 
        m *= beta1 
        m += (1 - beta1) * grad 
        v *= beta2 
        v += (1 - beta2) * grad * grad 
        mb = m / (1 - beta1 ** self.iter)
        vb = v / (1 - beta2 ** self.iter)
        param.data -= lr * mb / (np.sqrt(vb) + 1e-7)

class Nesterov(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr 
        self.momentum = momentum
        self.vs = {}

    def updateOne(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        
        v = self.vs[v_key]
        v *= self.momentum 
        v -= self.lr * param.grad.data 
        param.data += self.momentum * self.momentum * v 
        param.data -= (1 + self.momentum) * self.lr * param.grad.data
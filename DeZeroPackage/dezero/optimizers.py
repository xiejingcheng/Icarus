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


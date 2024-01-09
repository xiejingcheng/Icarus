import numpy as np
from utils import as_array

class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data
        self.creator = None
        self.grad = None

    def setCreator(self, func):
        self.creator = func

    # def backward(self):
    #     if self.grad is None:
    #         self.grad = np.ones_like(self.data)

    #     f = self.creator
    #     if f is not None:
    #         x = f.input
    #         x.grad = f.backward(self.grad)
    #         x.backward()

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        func = [self.creator]
        while func:
            f = func.pop()
            x = f.input
            y = f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                func.append(x.creator)

class Function:
    def __init__(self):
        self.output = None
        self.input = None

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        #设定指针
        output.setCreator(self)
        # print(output.creator)
        self.output = output
        self.input = input
        return output

    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, gy):
        raise NotImplementedError
import numpy as np
from core import Variable, Function

class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        return np.exp(self.inputs[0].data) * gy
    
class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    
    def backward(self, gy):
        return gy, gy
        
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

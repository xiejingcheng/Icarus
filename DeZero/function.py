import numpy as np
from core import Variable, Function

class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        return np.exp(self.input.data) * gy
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

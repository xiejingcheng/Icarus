import weakref
from icazero.core import Parameter
import icazero.functions as F
import numpy as np


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)
    
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outpus = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
    
class Linear(Layer):
    def __init__(self, outSize, nobias=False, dtype=np.float32, inSize=None):
        super().__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.inSize is not None:
            self._initW()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(outSize, dtype=dtype), name='b')

    def _initW(self):
        I, O = self.inSize, self.outSize
        Wdata = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1/I)
        self.W.data = Wdata

    def forward(self, x):
        if self.W.data is None:
            #todo 如果数据不是二维的怎么处理
            self.inSize = x.shape[1]
            self._initW()
        
        y = F.linear(x, self.W, self.b)
        return y
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.sigmoid(x)
    
class ReLU(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.relu(x)

class CNN(Layer):
    def __init__(self, outChannel, kernelSize, stride=1, pad=0, nobias=False):
        super().__init__()
        self.outChannel = outChannel
        self.kernelSize = kernelSize
        self.stride = stride
        self.pad = pad
        self.nobias = nobias
    
    def forward(self, x):
        return F.convolution2d(x, self.W, self.b, self.stride, self.pad)
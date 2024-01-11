import numpy as np
from utils import as_array
import weakref
import contextlib

class Config:
    enable_backprop = True

class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data
        self.creator = None
        self.grad = None
        self.generation = 0

    def setCreator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, ratain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seenSet = set()

        def addFunc(f):
            if f not in seenSet:
                funcs.append(f)
                seenSet.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        addFunc(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    addFunc(x.creator)

            if not ratain_grad:
                for y in f.outputs:
                    y().grad = None


class Function:
    def __init__(self):
        self.outputs = None
        self.inputs = None
        self.generation = 0

    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        #设定指针于代数
        #开启以后不会再引用，所有会被内存回收，最后CUDA实现的时候，可以手动删除
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.setCreator(self)
        
        self.outputs = [weakref.ref(output) for output in outputs]
        self.inputs = inputs
        return outputs if len(outputs) > 1 else outputs[0]

    
    def forward(self, xs):
        raise NotImplementedError
    
    def backward(self, gys):
        raise NotImplementedError

@contextlib.contextmanager 
def usingConfig(name, value):
    oldValue = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, oldValue)

def noGrad():
    return usingConfig('enable_backprop', False)
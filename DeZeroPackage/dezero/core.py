import numpy as np
import weakref
import contextlib
import dezero

class Config:
    enableBackprop = True

class Variable:
    __array_priority__ = 200

    def __init__(self, data, name = None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data
        self.creator = None
        self.grad = None
        self.name = name
        self.generation = 0
    
    @property
    def shape(self):
        return self.data.shape  
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return Variable(np.transpose(self.data))

    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def setName(self, name):
        self.name = name

    def setCreator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    
    def sum(self, axis=None, keepdims = False):
        return dezero.functions.sum(self, axis, keepdims)

    def backward(self, ratainGrad=False, createGraph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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

            with usingConfig('enableBackprop', createGraph):
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

            if not ratainGrad:
                for y in f.outputs:
                    y().grad = None

class Function:
    def __init__(self):
        self.outputs = None
        self.inputs = None
        self.generation = 0

    #call的时候forward的是data就是nd实例
    def __call__(self, *inputs):
        inputs = [asVariable(x) for x in inputs]
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(asArray(y)) for y in ys]

        if Config.enableBackprop:
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

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sumTo(gx0, self.x0_shape)
            gx1 = dezero.functions.sumTo(gx1, self.x1_shape)
        return gx0, gx1

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    
    def backward(self, gy):
        return gy, -gy  

class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy / x1, gy * (-x0 / x1**2)

class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    
def add(x0, x1):
    x1 = asArray(x1)
    return Add()(x0, x1)

def pow(x, c):
    return Pow(c)(x)

def div(x0, x1):
    x1 = asArray(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = asArray(x1)
    return Div()(x1, x0)

def rsub(x0, x1):
    x1 = asArray(x1)
    return Sub()(x1, x0)

def sub(x0, x1):
    x1 = asArray(x1)
    return Sub()(x0, x1)    
    
def neg(x):
    return Neg()(x)

def mul(x0, x1):
    x1 = asArray(x1)
    return Mul()(x0, x1) 

@contextlib.contextmanager 
def usingConfig(name, value):
    oldValue = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, oldValue)

def noGrad():
    return usingConfig('enableBackprop', False)

def asVariable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def asArray(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def setupVariable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow

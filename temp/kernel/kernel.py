import ctypes as ct
import numpy as np
from typing import Union
import sys
import weakref

from kernel.lib_load import Matrix, CUDAKernelException, LibLoader

lib_path = "./build/libica.so"

_lib = LibLoader(lib_path)

class Config:
    enableBackprop = True

def as_ndarray(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (int, float)):
        return np.array([x], dtype=np.float32)
    elif isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float32)
    else:
        raise ValueError("Unsupported input type for conversion to ndarray")


class Tensor:
    __array_priority__ = 200

    def __init__(self, data: Union[int, float, list, tuple, np.ndarray] = None, name: str = None, device: str = "cuda"):
        self.creator = None
        self.grad = None
        self.name = name
        self.generation = 0

        self.device = device

        nd_array = as_ndarray(data)

        if nd_array is not None:
            self.ndim = nd_array.ndim
            self.shape = nd_array.shape
            self.dtype = nd_array.dtype
            self.data = []
            if nd_array.ndim <= 2:
                self.data.appen(_lib.build_matrix_from_array(np.atleast_2d(nd_array)))
            elif nd_array.ndim == 3:
                for i in range(nd_array.shape[0]):
                    self.data.append(_lib.build_matrix_from_array(np.atleast_2d(nd_array[i])))
            elif nd_array.ndim == 4:
                for i in range(nd_array.shape[0]):
                    self.data.append([])
                    for j in range(nd_array.shape[1]):
                        self.data[i].append(_lib.build_matrix_from_array(np.atleast_2d(nd_array[i, j])))
            elif nd_array.ndim == 5:
                for i in range(nd_array.shape[0]):
                    self.data.append([])
                    for j in range(nd_array.shape[1]):
                        self.data[i].append([])
                        for k in range(nd_array.shape[2]):
                            self.data[i][j].append(_lib.build_matrix_from_array(np.atleast_2d(nd_array[i, j, k])))
            else:
                raise ValueError("Only support 5-dim tensor at most.")
            
        if self.device == "cuda":
            self.toCuda()

    def apply_function_one_to_self(self, data1, data1_shape, func):
        if len(data1_shape) <= 2:
            data1[0] = func(data1[0])
        elif len(data1_shape) == 3:
            for i in range(data1_shape[0]):
                data1[i] = func(data1[i])
        elif len(data1_shape) == 4:
            for i in range(data1_shape[0]):
                for j in range(data1_shape[1]):
                    data1[i][j] = func(data1[i][j])
        elif len(data1_shape) == 5:
            for i in range(data1_shape[0]):
                for j in range(data1_shape[1]):
                    for k in range(data1_shape[2]):
                        data1[i][j][k] = func(data1[i][j][k])
        else:
            raise ValueError("Only support 5-dim tensor at most.")
        
    def apply_function_one_to_none(self, data1, data1_shape, func):
        if len(data1_shape) <= 2:
            func(data1[0])
        elif len(data1_shape) == 3:
            for i in range(data1_shape[0]):
                func(data1[i])
        elif len(data1_shape) == 4:
            for i in range(data1_shape[0]):
                for j in range(data1_shape[1]):
                    func(data1[i][j])
        elif len(data1_shape) == 5:
            for i in range(data1_shape[0]):
                for j in range(data1_shape[1]):
                    for k in range(data1_shape[2]):
                        func(data1[i][j][k])
        else:
            raise ValueError("Only support 5-dim tensor at most.")
    
    def apply_function_two_to_self(self, data1, data1_shape, data2, data2_shape, func):
        if len(data1_shape) <= 2:
            data1[0] = func(data1[0], data2[0])
        elif len(data1_shape) == 3:
            for i in range(data1_shape[0]):
                data1[i] = func(data1[i], data2[i])
        elif len(data1_shape) == 4:
            for i in range(data1_shape[0]):
                for j in range(data1_shape[1]):
                    data1[i][j] = func(data1[i][j], data2[i][j])
        elif len(data1_shape) == 5:
            for i in range(data1_shape[0]):
                for j in range(data1_shape[1]):
                    for k in range(data1_shape[2]):
                        data1[i][j][k] = func(data1[i][j][k], data2[i][j][k])
        else:
            raise ValueError("Only support 5-dim tensor at most.")

    
    def apply_function_one_to_one(data1, data1_shape, func):
        if len(data1_shape) <= 2:
            return [func(data1[0])]
        elif len(data1_shape) == 3:
            return [func(data1[i]) for i in range(data1_shape[0])]
        elif len(data1_shape) == 4:
            return [[func(data1[i][j]) for j in range(data1_shape[1])] for i in range(data1_shape[0])]
        elif len(data1_shape) == 5:
            return [[[func(data1[i][j][k]) for k in range(data1_shape[2])] for j in range(data1_shape[1])] for i in range(data1_shape[0])]
        else:
            raise ValueError("Only support 5-dim tensor at most.")
        
        
    def apply_function_two_to_one(self, data1, data1_shape, data2, data2_shape, func):
        if len(data1_shape) <= 2:
            return [func(data1[0], data2[0])]
        elif len(data1_shape) == 3:
            return [func(data1[i], data2[i]) for i in range(data1_shape[0])]
        elif len(data1_shape) == 4:
            return [[func(data1[i][j], data2[i][j]) for j in range(data1_shape[1])] for i in range(data1_shape[0])]
        elif len(data1_shape) == 5:
            return [[[func(data1[i][j][k], data2[i][j][k]) for k in range(data1_shape[2])] for j in range(data1_shape[1])] for i in range(data1_shape[0])]
        else:
            raise ValueError("Only support 5-dim tensor at most.")
        
    
    def toCuda(self):
        if self.device == "cuda":
            return
        
        self.apply_function_one_to_none(self.data, _lib.copy_to_device)
        self.device = "cuda"


    def toCpu(self):
        if self.device == "cpu":
            return
        if self.ndim <= 2:
            self.data[0] = _lib.copy_to_host(self.data[0])
        elif self.ndim == 3:
            for i in range(self.shape[0]):
                self.data[i] = _lib.copy_to_host(self.data[i])
        elif self.ndim == 4:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.data[i][j] = _lib.copy_to_host(self.data[i][j])
        elif self.ndim == 5:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[2]):
                        self.data[i][j][k] = _lib.copy_to_host(self.data[i][j][k])
        else:
            raise ValueError("Only support 5-dim tensor at most.")
        
        self.device = "cpu"
        
    def setName(self, name):
        self.name = name

    def setCreator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    
    def backward(self, ratainGrad=False, createGraph=False):
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.shape))

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

            if not ratainGrad:
                for y in f.outputs:
                    y().grad = None


class Function:
    def __init__(self):
        self.outputs = None
        self.inputs = None
        self.generation = 0

    def __call__(self, *inputs):
        inputs = [asTensor(x) for x in inputs]
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Tensor(data = as_ndarray(y)) for y in ys]

        if Config.enableBackprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.setCreator(self)
        
        self.outputs = [weakref.ref(output) for output in outputs]
        self.inputs = inputs
        return outputs if len(outputs) > 1 else outputs[0]
    
def asTensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(data = x)

            

                

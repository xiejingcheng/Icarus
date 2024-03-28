import ctypes as ct
import numpy as np
from typing import Union


class Tensor(ct.Structure):
    _fields_ = [
        ("data_host", ct.POINTER(ct.c_float)),
        ("data_device", ct.POINTER(ct.c_float)),
        ("on_device", ct.c_int),
        ("on_host", ct.c_int),
        ("dims", ct.c_int),
        ("size", ct.c_int * 10),
        ("is_trans", ct.c_int),
        ("owns_data", ct.c_int)
    ]

class LibLoader():
    def __init__(self,lib_path:str):
        self.lib = ct.cdll.LoadLibrary(lib_path)
    
    def print_device(self):
        self.lib.print_device()

    def build_matrix_empty(self, m:int, n:int):
        mat = Tensor()
        self.lib.build_matrix_empty(ct.c_int(m), ct.c_int(n), ct.pointer(mat))
        return mat
    
    #这里的参数接受应该优化一下
    # def build_tensor_empty(self, dims:int, size:Union[int, list]):
    #     data = Tensor()
    #     self.lib.build_tensor_empty(dims, size, ct.pointer(data))
    #     return data
    
    def copy_to_device(self, data:Tensor):
        self.lib.copy_to_device(ct.pointer(data))

    def copy_to_host(self, data:Tensor):
        self.lib.copy_to_host(ct.pointer(data))

    def print_tensor(self, data:Tensor):
        self.lib.print_tensor(ct.pointer(data))
    
    def gemm_on_device(self, A: Tensor, B: Tensor):
        C = self.build_matrix_empty(A.size[0], B.size[1])
        self.copy_to_device(C)
        self.lib.gemm_on_device(ct.pointer(A), ct.pointer(B), ct.pointer(C))
        return C 


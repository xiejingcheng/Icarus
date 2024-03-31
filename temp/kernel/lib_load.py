import ctypes as ct
import numpy as np
from typing import Union
import sys

lib_path = "/h3cstore_ns/jcxie/Icarus/kernel/build/libica.so"
_lib = ct.cdll.LoadLibrary(lib_path)

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

class CUDAKernelException(Exception):
    pass

def get_last_cuda_error():
        errmsg = _lib.get_last_cuda_error()
        if sys.version_info >= (3,):
            return bytes(errmsg).decode()
        else:
            return str(errmsg)

def generate_exception(err_code, **kwargs):

    if err_code == -1:
        return CUDAKernelException("Incompatible matrix dimensions.")
    elif err_code == -2:
        return CUDAKernelException("CUBLAS error.")
    elif err_code == -3:
        return CUDAKernelException("CUDA error: " + get_last_cuda_error())
    elif err_code == -4:
        return CUDAKernelException("Operation not supported on views.")
    elif err_code == -5:
        return CUDAKernelException("Operation not supported on "
                                "transposed matrices.")
    elif err_code == -6:
        return CUDAKernelException("")
    elif err_code == -7:
        return CUDAKernelException("Incompatible transposedness.")
    elif err_code == -8:
        return CUDAKernelException("Matrix is not in device memory.")
    elif err_code == -9:
        return CUDAKernelException("Operation not supported.")
    # elif err_code == -10:
    #     filepath = kwargs.get("filepath","");
    #     if filepath:
    #         filepath = ": '%s'" % filepath
    #     return CUDAKernelException("Cannot open file%s: %s" % (filepath,get_last_clib_error()))
    elif err_code == -11:
        filepath = kwargs.get("filepath","");
        if filepath:
            filepath = ": '%s'" % filepath
        return CUDAKernelException("Cannot parse file%s." % filepath)
    elif err_code == -99:
        return CUDAKernelException("ERROR_INCOMLETE.")
    else:
        return CUDAKernelException("")

class LibLoader():
    def __init__(self):
        self.lib = _lib
    
    def print_device(self):
        self.lib.print_device()

    def build_matrix_empty(self, m:int, n:int):
        mat = Tensor()
        err_code = self.lib.build_matrix_empty(ct.c_int(m), ct.c_int(n), ct.pointer(mat))
        if err_code:
            raise CUDAKernelException(err_code)
        return mat
    
    #这里的参数接受应该优化一下
    # def build_tensor_empty(self, dims:int, size:Union[int, list]):
    #     data = Tensor()
    #     self.lib.build_tensor_empty(dims, size, ct.pointer(data))
    #     return data
    
    def copy_to_device(self, data:Tensor):
        err_code = self.lib.copy_to_device(ct.pointer(data))
        if err_code:
            raise generate_exception(err_code)

    def copy_to_host(self, data:Tensor):
        err_code = self.lib.copy_to_host(ct.pointer(data))
        if err_code:
            raise generate_exception(err_code)

    def print_tensor(self, data:Tensor):
        self.lib.print_tensor(ct.pointer(data))
    
    def gemm_on_device(self, A: Tensor, B: Tensor):
        C = self.build_matrix_empty(A.size[0], B.size[1])
        self.copy_to_device(C)
        err_code = self.lib.gemm_on_device(ct.pointer(A), ct.pointer(B), ct.pointer(C))
        if err_code:
            raise generate_exception(err_code)
        return C 


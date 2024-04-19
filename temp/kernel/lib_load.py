import ctypes as ct
import numpy as np
from typing import Union
import sys

lib_path = "./build/libica.so"


class Matrix(ct.Structure):
    _fields_ = [
        ("data_host", ct.POINTER(ct.c_float)),
        ("data_device", ct.POINTER(ct.c_float)),
        ("on_device", ct.c_int),
        ("on_host", ct.c_int),
        ("dims", ct.c_int),
        ("size", ct.c_int * 2),
        ("is_trans", ct.c_int),
        ("owns_data", ct.c_int)
    ]

class CUDAKernelException(Exception):
    pass


class LibLoader():
    def __init__(self, lib_path):
        self.lib = ct.cdll.LoadLibrary(lib_path)

    def get_last_cuda_error(self):
        errmsg = self.lib.get_last_cuda_error()
        if sys.version_info >= (3,):
            return bytes(errmsg).decode()
        else:
            return str(errmsg)

    def get_last_clib_error(self):
        errmsg = self.lib.get_last_clib_error()
        if sys.version_info >= (3,):
            return bytes(errmsg).decode()
        else:
            return str(errmsg)

    def generate_exception(self, err_code, **kwargs):

        if err_code == -1:
            return CUDAKernelException("Incompatible matrix dimensions.")
        elif err_code == -2:
            return CUDAKernelException("CUBLAS error.")
        elif err_code == -3:
            return CUDAKernelException("CUDA error: " + self.get_last_cuda_error())
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
        elif err_code == -10:
            filepath = kwargs.get("filepath","")
            if filepath:
                filepath = ": '%s'" % filepath
            return CUDAKernelException("Cannot open file%s: %s" % (filepath, self.get_last_clib_error()))
        elif err_code == -11:
            filepath = kwargs.get("filepath","")
            if filepath:
                filepath = ": '%s'" % filepath
            return CUDAKernelException("Cannot parse file%s." % filepath)
        elif err_code == -99:
            return CUDAKernelException("ERROR_INCOMLETE.")
        else:
            return CUDAKernelException("")
    
    def print_device(self):
        self.lib.print_device()

    def cuda_set_device(self, device:int):
        err_code = self.lib.cuda_set_device(ct.c_int(device))
        if err_code:
            raise self.generate_exception(err_code)
    
    def set_trans(self, data:Matrix, is_trans:int):
        self.lib.set_transpose(ct.pointer(data), ct.c_int(is_trans))

    def cuda_sync_threads(self):
        self.lib.cuda_sync_threads()

    def allocate_device_memory(self, data:Matrix):
        err_code = self.lib.allocate_device_memory(ct.pointer(data))
        if err_code:
            raise self.generate_exception(err_code)

    def copy_to_host(self, data:Matrix):
        err_code = self.lib.copy_to_host(ct.pointer(data))
        if err_code:
            raise self.generate_exception(err_code)

    def copy_to_device(self, data:Matrix):
        err_code = self.lib.copy_to_device(ct.pointer(data))
        if err_code:
            raise self.generate_exception(err_code)

    def copy_on_device(self, A:Matrix, B:Matrix):
        err_code = self.lib.copy_on_device(ct.pointer(A), ct.pointer(B))
        if err_code:
            raise self.generate_exception(err_code)

    def free_device_memory(self, data:Matrix):
        err_code = self.lib.free_device_memory(ct.pointer(data))
        if err_code:
            raise self.generate_exception(err_code)

    def build_matrix_empty_host(self, m:int, n:int):
        mat = Matrix()
        err_code = self.lib.build_matrix_empty_host(ct.c_int(m), ct.c_int(n), ct.pointer(mat))
        if err_code:
            raise CUDAKernelException(err_code)
        return mat
    
    def build_matrix_empty_device(self, m:int, n:int):
        mat = Matrix()
        err_code = self.lib.build_matrix_empty_device(ct.c_int(m), ct.c_int(n), ct.pointer(mat))
        if err_code:
            raise CUDAKernelException(err_code)
        return mat

    def build_matrix_from_array(self, data:np.ndarray):
        m, n = data.shape
        mat = Matrix()
        err_code = self.lib.build_matrix_from_array(ct.c_int(m), ct.c_int(n), ct.pointer(mat), data.ctypes.data_as(ct.POINTER(ct.c_float)))
        if err_code:
            raise CUDAKernelException(err_code)
        return mat
    
    def from_matrix_to_array(self, data:Matrix):
        m, n = data.size
        arr = np.zeros((m, n), dtype=np.float32)
        err_code = self.lib.from_matrix_to_array(ct.pointer(data), arr.ctypes.data_as(ct.POINTER(ct.c_float)))
        if err_code:
            raise self.generate_exception(err_code)
        return arr
    
    def copy_to_device(self, data:Matrix):
        err_code = self.lib.copy_to_device(ct.pointer(data))
        if err_code:
            raise self.generate_exception(err_code)

    def copy_to_host(self, data:Matrix):
        err_code = self.lib.copy_to_host(ct.pointer(data))
        if err_code:
            raise self.generate_exception(err_code)

    def print_Matrix(self, data:Matrix):
        self.lib.print_Matrix(ct.pointer(data))

    def fourop_on_device(self, A:Matrix, B:Matrix, C:Matrix, op:int):
        err_code = self.lib.fourop_on_device(ct.pointer(A), ct.pointer(B), ct.pointer(C), ct.c_int(op))
        if err_code:
            raise self.generate_exception(err_code)
    
    def gemm_on_device(self, A: Matrix, B: Matrix):
        C = self.build_matrix_empty_device(A.size[0], B.size[1])
        err_code = self.lib.gemm_on_device(ct.pointer(A), ct.pointer(B), ct.pointer(C))
        if err_code:
            raise self.generate_exception(err_code)
        return C
        
    def sigmod_on_device(self, A: Matrix, B: Matrix):
        err_code = self.lib.sigmod_on_device(ct.pointer(A), ct.pointer(B))
        if err_code:
            raise self.generate_exception(err_code)
        



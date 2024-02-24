import ctypes as ct
import numpy as np
from typing import Union

class CuCuMatException(Exception):
    pass

def generate_exception(succ: bool):
    if not succ:
        raise CuCuMatException()

class CMatrix(ct.Structure):
    _fields_ = [("m", ct.c_int),
                ("n", ct.c_int),
                ("data", ct.POINTER(ct.c_float))]

class LibWrap:
    def __init__(self, lib_path: str):
        self.lib = ct.cdll.LoadLibrary(lib_path)
        self._cublas_create()
    
    def __del__(self):
        self._cublas_destroy()
        
# ---------------------------- C API ----------------------------

    def _cublas_create(self):
        succ = self.lib.cublas_create()
        generate_exception(succ)
    
    def _cublas_destroy(self):
        if hasattr(self, "lib"):
            succ = self.lib.cublas_destroy()
            generate_exception(succ)
    
    def build_matrix_empty(self, m: int, n: int, cmat: CMatrix):
        succ = self.lib.build_matrix_empty(ct.c_int(m), ct.c_int(n), ct.pointer(cmat))
        generate_exception(succ)
    
    def build_matrix_zeros(self, m: int, n: int, cmat: CMatrix):
        succ = self.lib.build_matrix_with_fill(ct.c_int(m), ct.c_int(n), ct.pointer(cmat), ct.c_float(0.0))
        generate_exception(succ)
    
    def build_matrix_ones(self, m: int, n: int, cmat: CMatrix):
        succ = self.lib.build_matrix_with_fill(ct.c_int(m), ct.c_int(n), ct.pointer(cmat), ct.c_float(1.0))
        generate_exception(succ)
    
    def build_matrix_from_array(self, cmat: CMatrix, arr: np.ndarray):
        arr = np.array(arr, dtype=np.float32, order="F")
        succ = self.lib.build_matrix_from_array(ct.c_int(arr.shape[0]),
                                                ct.c_int(arr.shape[1]),
                                                ct.pointer(cmat),
                                                arr.ctypes.data_as(ct.POINTER(ct.c_float)))
        generate_exception(succ)
    
    def to_host(self, cmat: CMatrix) -> np.ndarray:
        arr = np.empty((cmat.m, cmat.n), dtype=np.float32, order="F")
        succ = self.lib.to_host(ct.pointer(cmat), arr.ctypes.data_as(ct.POINTER(ct.c_float)))
        generate_exception(succ)
        return arr
    
    def free_device_memory(self, cmat: CMatrix):
        succ = self.lib.free_device_memory(ct.pointer(cmat))
        generate_exception(succ)
        
    def assign(self, cmat: CMatrix, source: Union[CMatrix, np.ndarray]):
        if isinstance(source, CMatrix):
            succ = self.lib.assign(ct.pointer(cmat),
                                   source.data,
                                   ct.c_bool(False))
        else:
            arr = np.array(source, dtype=np.float32, order="F")
            succ = self.lib.assign(ct.pointer(cmat),
                                   arr.ctypes.data_as(ct.POINTER(ct.c_float)),
                                   ct.c_bool(True))
        generate_exception(succ)
    
    def fill(self, cmat: CMatrix, val: float):
        succ = self.lib.fill(ct.pointer(cmat), ct.c_float(val))
        generate_exception(succ)
    
    def copy(self, src: CMatrix, dst: CMatrix):
        succ = self.lib.copy(ct.pointer(src), ct.pointer(dst))
        generate_exception(succ)
    
    def abs(self, src: CMatrix, dst: CMatrix):
        succ = self.lib.matrix_abs(ct.pointer(src), ct.pointer(dst))
        generate_exception(succ)

    def negative(self, src: CMatrix, dst: CMatrix):
        succ = self.lib.matrix_negative(ct.pointer(src), ct.pointer(dst))
        generate_exception(succ)
    
    def reciprocal(self, src: CMatrix, dst: CMatrix):
        succ = self.lib.matrix_reciprocal(ct.pointer(src), ct.pointer(dst))
        generate_exception(succ)
        
    def add(self, src: CMatrix, dst: CMatrix, val: Union[CMatrix, float]):
        if isinstance(val, CMatrix):
            succ = self.lib.matrix_add_matrix(ct.pointer(src),
                                              ct.pointer(dst),
                                              ct.pointer(val))
        else:
            succ = self.lib.matrix_add_scalar(ct.pointer(src),
                                              ct.pointer(dst),
                                              ct.c_float(val))
        generate_exception(succ)
    
    def sub(self, src: CMatrix, dst: CMatrix, val: Union[CMatrix, float]):
        if isinstance(val, CMatrix):
            succ = self.lib.matrix_sub_matrix(ct.pointer(src),
                                              ct.pointer(dst),
                                              ct.pointer(val))
        else:
            succ = self.lib.matrix_sub_scalar(ct.pointer(src),
                                              ct.pointer(dst),
                                              ct.c_float(val))
        generate_exception(succ)
    
    def mul(self, src: CMatrix, dst: CMatrix, val: Union[CMatrix, float]):
        if isinstance(val, CMatrix):
            succ = self.lib.matrix_mul_matrix(ct.pointer(src),
                                              ct.pointer(dst),
                                              ct.pointer(val))
        else:
            succ = self.lib.matrix_mul_scalar(ct.pointer(src),
                                              ct.pointer(dst),
                                              ct.c_float(val))
        generate_exception(succ)
        
    def div(self, src: CMatrix, dst: CMatrix, val: Union[CMatrix, float]):
        if isinstance(val, CMatrix):
            succ = self.lib.matrix_div_matrix(ct.pointer(src),
                                              ct.pointer(dst),
                                              ct.pointer(val))
        else:
            succ = self.lib.matrix_div_scalar(ct.pointer(src),
                                              ct.pointer(dst),
                                              ct.c_float(val))
        generate_exception(succ)
    
    def view_cols(self, src: CMatrix, dst: CMatrix, start: int, end: int):
        succ = self.lib.view_cols(ct.pointer(src),
                                  ct.pointer(dst),
                                  ct.c_int(start),
                                  ct.c_int(end))
        generate_exception(succ)
    
    def get_rows(self, src: CMatrix, dst: CMatrix, start: int, end: int):
        succ = self.lib.get_rows(ct.pointer(src),
                                 ct.pointer(dst),
                                 ct.c_int(start),
                                 ct.c_int(end))
        generate_exception(succ)
    
    def set_rows(self, src: CMatrix, dst: CMatrix, start: int, end: int):
        succ = self.lib.set_rows(ct.pointer(src),
                                 ct.pointer(dst),
                                 ct.c_int(start),
                                 ct.c_int(end))
        generate_exception(succ)
    
    def transpose(self, src: CMatrix, dst: CMatrix):
        succ = self.lib.transpose(ct.pointer(src), ct.pointer(dst))
        generate_exception(succ)
    
    def axis_zero_sum(self, src: CMatrix, dst: CMatrix):
        succ = self.lib.axis_zero_sum(ct.pointer(src), ct.pointer(dst))
        generate_exception(succ)
    
    def all_sum(self, cmat: CMatrix) -> float:
        result = ct.c_float(0.0)
        succ = self.lib.all_sum(ct.pointer(cmat), ct.pointer(result))
        generate_exception(succ)
        return result.value
    
    def less_than(self, src: CMatrix, dst: CMatrix, val: Union[CMatrix, float]):
        if isinstance(val, CMatrix):
            succ = self.lib.less_than_matrix(ct.pointer(src),
                                             ct.pointer(dst),
                                             ct.pointer(val))
        else:
            succ = self.lib.less_than_scalar(ct.pointer(src),
                                             ct.pointer(dst),
                                             ct.c_float(val))
        generate_exception(succ)
    
    def greater_than(self, src: CMatrix, dst: CMatrix, val: Union[CMatrix, float]):
        if isinstance(val, CMatrix):
            succ = self.lib.greater_than_matrix(ct.pointer(src),
                                                ct.pointer(dst),
                                                ct.pointer(val))
        else:
            succ = self.lib.greater_than_scalar(ct.pointer(src),
                                                ct.pointer(dst),
                                                ct.c_float(val))
        generate_exception(succ)
    
    def equal_to(self, src: CMatrix, dst: CMatrix, val: Union[CMatrix, float]):
        if isinstance(val, CMatrix):
            succ = self.lib.equal_to_matrix(ct.pointer(src),
                                            ct.pointer(dst),
                                            ct.pointer(val))
        else:
            succ = self.lib.equal_to_scalar(ct.pointer(src),
                                            ct.pointer(dst),
                                            ct.c_float(val))
        generate_exception(succ)
    
    def axis_zero_min(self, src: CMatrix, dst: CMatrix):
        succ = self.lib.axis_zero_min(ct.pointer(src), ct.pointer(dst))
        generate_exception(succ)
    
    def all_min(self, cmat: CMatrix) -> float:
        result = ct.c_float(0.0)
        succ = self.lib.all_min(ct.pointer(cmat), ct.pointer(result))
        generate_exception(succ)
        return result.value
    
    def axis_zero_max(self, src: CMatrix, dst: CMatrix):
        succ = self.lib.axis_zero_max(ct.pointer(src), ct.pointer(dst))
        generate_exception(succ)
    
    def all_max(self, cmat: CMatrix) -> float:
        result = ct.c_float(0.0)
        succ = self.lib.all_max(ct.pointer(cmat), ct.pointer(result))
        generate_exception(succ)
        return result.value
    
    def dot(self, mat1: CMatrix, mat2: CMatrix, target_mat: CMatrix):
        succ = self.lib.dot(ct.pointer(mat1), ct.pointer(mat2), ct.pointer(target_mat))
        generate_exception(succ)
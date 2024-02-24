import numpy as np
import os
from lib_wrap import LibWrap, CMatrix
from typing import Union, Tuple, Optional

_lib = LibWrap(os.path.join(os.path.dirname(__file__), ".\cuda\cucumat.lib"))

class Matrix:
    def __init__(self, cmat: CMatrix, own: bool = True):
        self.cmat = cmat
        self.own = own
    
    def __del__(self):
        if self.own:
            _lib.free_device_memory(self.cmat)

    @classmethod
    def matrix_empty(cls, m: int, n: int) -> 'Matrix':
        cmat = CMatrix()
        _lib.build_matrix_empty(m, n, cmat)
        return cls(cmat)
    
    @classmethod
    def matrix_zeros(cls, m: int, n: int) -> 'Matrix':
        cmat = CMatrix()
        _lib.build_matrix_zeros(m, n, cmat)
        return cls(cmat)
    
    @classmethod
    def matrix_ones(cls, m: int, n: int) -> 'Matrix':
        cmat = CMatrix()
        _lib.build_matrix_ones(m, n, cmat)
        return cls(cmat)
    
    @classmethod
    def matrix_from_array(cls, arr: np.ndarray) -> 'Matrix':
        cmat = CMatrix()
        _lib.build_matrix_from_array(cmat, arr)
        return cls(cmat)
    
    def to_numpy(self) -> np.ndarray:
        return _lib.to_host(self.cmat)

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.cmat.m), int(self.cmat.n))
    
    def reshape(self, m: int, n: int):
        self.cmat.m = m
        self.cmat.n = n
        
    def assign(self, mat: Union['Matrix', np.ndarray]):
        if isinstance(mat, Matrix):
            _lib.assign(self.cmat, mat.cmat)
        else:
            _lib.assign(self.cmat, mat)
    
    def fill(self, val: float):
        _lib.fill(self.cmat, val)
    
    def copy(self) -> 'Matrix':
        cmat = CMatrix()
        _lib.copy(self.cmat, cmat)
        return self.__class__(cmat)
    
    def _build_target_mat(self, inplace: bool) -> 'Matrix':
        target_mat = self
        if not inplace:
            target_mat = Matrix.matrix_empty(*self.shape)
        return target_mat
    
    def abs(self, inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        _lib.abs(self.cmat, target_mat.cmat)
        return target_mat

    def negative(self, inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        _lib.negative(self.cmat, target_mat.cmat)
        return target_mat
    
    def reciprocal(self, inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        _lib.reciprocal(self.cmat, target_mat.cmat)
        return target_mat
    
    def add(self, val: Union['Matrix', float], inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        if isinstance(val, Matrix):
            _lib.add(self.cmat, target_mat.cmat, val.cmat)
        else:
            _lib.add(self.cmat, target_mat.cmat, val)
        return target_mat
    
    def sub(self, val: Union['Matrix', float], inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        if isinstance(val, Matrix):
            _lib.sub(self.cmat, target_mat.cmat, val.cmat)
        else:
            _lib.sub(self.cmat, target_mat.cmat, val)
        return target_mat
    
    def mul(self, val: Union['Matrix', float], inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        if isinstance(val, Matrix):
            _lib.mul(self.cmat, target_mat.cmat, val.cmat)
        else:
            _lib.mul(self.cmat, target_mat.cmat, val)
        return target_mat
    
    def div(self, val: Union['Matrix', float], inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        if isinstance(val, Matrix):
            _lib.div(self.cmat, target_mat.cmat, val.cmat)
        else:
            _lib.div(self.cmat, target_mat.cmat, val)
        return target_mat
    
    def dot(self, mat: 'Matrix') -> 'Matrix':
        # [m, k] dot [k, n] ==> [m, n]
        m, k, n = self.shape[0], self.shape[1], mat.shape[1]
        target_mat = Matrix.matrix_zeros(m, n)
        _lib.dot(self.cmat, mat.cmat, target_mat.cmat)
        return target_mat
    
    def get_cols(self, start: int, end: int, view: bool = False) -> 'Matrix':
        view_cmat = CMatrix()
        _lib.view_cols(self.cmat, view_cmat, start, end)
        if view:
            return Matrix(view_cmat, own=False)
        copy_cmat = CMatrix()
        _lib.copy(view_cmat, copy_cmat)
        return Matrix(copy_cmat)
    
    def set_cols(self, start: int, end: int, mat: 'Matrix'):
        self.get_cols(start, end, view=True).assign(mat)
    
    def get_rows(self, start: int, end: int) -> 'Matrix':
        copy_cmat = CMatrix()
        _lib.get_rows(self.cmat, copy_cmat, start, end)
        return Matrix(copy_cmat)
    
    def set_rows(self, start: int, end: int, mat: 'Matrix'):
        _lib.set_rows(mat.cmat, self.cmat, start, end)
    
    def transpose(self, inplace: bool = True) -> 'Matrix':
        m, n = self.shape
        target_mat = Matrix.matrix_empty(n, m)
        _lib.transpose(self.cmat, target_mat.cmat)
        if inplace:
            self.reshape(n, m)
            self.assign(target_mat)
            return self
        else:
            return target_mat
    
    def sum(self, axis: Optional[int] = None) -> Union[float, 'Matrix']:
        if axis == 0:
            target_mat = Matrix.matrix_empty(1, self.shape[1])
            _lib.axis_zero_sum(self.cmat, target_mat.cmat)
            return target_mat
        elif axis == 1:
            src_mat = self.transpose(inplace=False)
            dst_mat = Matrix.matrix_empty(1, src_mat.shape[1])
            _lib.axis_zero_sum(src_mat.cmat, dst_mat.cmat)
            dst_mat.reshape(src_mat.shape[1], 1)
            return dst_mat
        else:
            return _lib.all_sum(self.cmat)
    
    def mean(self, axis: Optional[int] = None) -> Union[float, 'Matrix']:
        if axis == 0:
            target_mat = self.sum(axis).div(self.shape[0])
            return target_mat
        elif axis == 1:
            target_mat = self.sum(axis).div(self.shape[1])
            return target_mat
        else:
            return self.sum(axis) / (self.shape[0] * self.shape[1])
        
    def min(self, axis: Optional[int] = None) -> Union[float, 'Matrix']:
        if axis == 0:
            target_mat = Matrix.matrix_empty(1, self.shape[1])
            _lib.axis_zero_min(self.cmat, target_mat.cmat)
            return target_mat
        elif axis == 1:
            src_mat = self.transpose(inplace=False)
            dst_mat = Matrix.matrix_empty(1, src_mat.shape[1])
            _lib.axis_zero_min(src_mat.cmat, dst_mat.cmat)
            dst_mat.reshape(src_mat.shape[1], 1)
            return dst_mat
        else:
            return _lib.all_min(self.cmat)
    
    def max(self, axis: Optional[int] = None) -> Union[float, 'Matrix']:
        if axis == 0:
            target_mat = Matrix.matrix_empty(1, self.shape[1])
            _lib.axis_zero_max(self.cmat, target_mat.cmat)
            return target_mat
        elif axis == 1:
            src_mat = self.transpose(inplace=False)
            dst_mat = Matrix.matrix_empty(1, src_mat.shape[1])
            _lib.axis_zero_max(src_mat.cmat, dst_mat.cmat)
            dst_mat.reshape(src_mat.shape[1], 1)
            return dst_mat
        else:
            return _lib.all_max(self.cmat)
    
    def less_than(self, val: Union['Matrix', float], inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        if isinstance(val, Matrix):
            _lib.less_than(self.cmat, target_mat.cmat, val.cmat)
        else:
            _lib.less_than(self.cmat, target_mat.cmat, val)
        return target_mat
    
    def greater_than(self, val: Union['Matrix', float], inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        if isinstance(val, Matrix):
            _lib.greater_than(self.cmat, target_mat.cmat, val.cmat)
        else:
            _lib.greater_than(self.cmat, target_mat.cmat, val)
        return target_mat
    
    def equal_to(self, val: Union['Matrix', float], inplace: bool = True) -> 'Matrix':
        target_mat = self._build_target_mat(inplace)
        if isinstance(val, Matrix):
            _lib.equal_to(self.cmat, target_mat.cmat, val.cmat)
        else:
            _lib.equal_to(self.cmat, target_mat.cmat, val)
        return target_mat
    
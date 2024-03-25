import unittest
from cucumat import Matrix
import numpy as np

class TestMatrix(unittest.TestCase):
    def test_matrix_empty(self):
        m, n = np.random.randint(100, 1000, (2,))
        mat = Matrix.matrix_empty(m, n)
        self.assertEqual(m, mat.shape[0])
        self.assertEqual(n, mat.shape[1])
    
    def test_matrix_zeros(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.zeros((m, n), dtype=np.float32)
        arr2 = Matrix.matrix_zeros(m, n).to_numpy()
        self.assertTrue(np.array_equal(arr1, arr2))
        
    def test_matrix_ones(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.ones((m, n), dtype=np.float32)
        arr2 = Matrix.matrix_ones(m, n).to_numpy()
        self.assertTrue(np.array_equal(arr1, arr2))
    
    def test_matrix_from_array(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = Matrix.matrix_from_array(arr1).to_numpy()
        self.assertTrue(np.array_equal(arr1, arr2))

    def test_shape(self):
        m, n = np.random.randint(100, 1000, (2,))
        mat = Matrix.matrix_zeros(m, n)
        self.assertEqual(m, mat.shape[0])
        self.assertEqual(n, mat.shape[1])
    
    def test_reshape(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        mat = Matrix.matrix_from_array(arr1)
        mat.reshape(n, m)
        arr2 = mat.to_numpy()
        arr1 = arr1.reshape(n, m, order="F")
        self.assertTrue(np.array_equal(arr1, arr2))
        
    def test_assign_from_Matrix(self):
        m, n = np.random.randint(100, 1000, (2,))
        mat1 = Matrix.matrix_ones(m, n)
        mat2 = Matrix.matrix_zeros(m, n)
        mat2.assign(mat1)
        arr1 = mat1.to_numpy()
        arr2 = mat2.to_numpy()
        self.assertTrue(np.array_equal(arr1, arr2))
    
    def test_assign_from_ndarray(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.rand(m, n).astype(np.float32)
        mat = Matrix.matrix_zeros(m, n)
        mat.assign(arr1)
        arr2 = mat.to_numpy()
        self.assertTrue(np.array_equal(arr1, arr2))

    def test_fill(self):
        m, n = np.random.randint(100, 1000, (2,))
        val = np.random.randn()
        arr1 = np.full((m, n), val, dtype=np.float32)
        mat = Matrix.matrix_zeros(m, n)
        mat.fill(val)
        arr2 = mat.to_numpy()
        self.assertTrue(np.array_equal(arr1, arr2))
    
    def test_copy(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.rand(m, n).astype(np.float32)
        mat1 = Matrix.matrix_zeros(m, n)
        mat1.assign(arr1)
        mat2 = mat1.copy()
        arr2 = mat2.to_numpy()
        self.assertTrue(np.array_equal(arr1, arr2))
    
    def test_abs(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = Matrix.matrix_from_array(arr1).abs().to_numpy()
        arr3 = Matrix.matrix_from_array(arr1).abs(inplace=False).to_numpy()
        arr1 = np.abs(arr1)
        self.assertTrue(np.array_equal(arr1, arr2))
        self.assertTrue(np.array_equal(arr1, arr3))
    
    def test_negative(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = Matrix.matrix_from_array(arr1).negative().to_numpy()
        arr3 = Matrix.matrix_from_array(arr1).negative(inplace=False).to_numpy()
        arr1 = -arr1
        self.assertTrue(np.array_equal(arr1, arr2))
        self.assertTrue(np.array_equal(arr1, arr3))
    
    def test_reciprocal(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = Matrix.matrix_from_array(arr1).reciprocal().to_numpy()
        arr3 = Matrix.matrix_from_array(arr1).reciprocal(inplace=False).to_numpy()
        arr1 = 1.0 / arr1
        self.assertTrue(np.array_equal(arr1, arr2))
        self.assertTrue(np.array_equal(arr1, arr3))
    
    def test_matrix_add_matrix(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = np.random.randn(m, n).astype(np.float32)
        arr3 = arr1 + arr2
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = Matrix.matrix_from_array(arr2)
        mat3 = mat1.add(mat2, inplace=False)
        mat1.add(mat2)
        
        self.assertTrue(np.array_equal(arr3, mat3.to_numpy()))
        self.assertTrue(np.array_equal(arr3, mat1.to_numpy()))
    
    def test_matrix_add_scalar(self):
        m, n = np.random.randint(100, 1000, (2,))
        val = np.random.randn()
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1 + val
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.add(val, inplace=False)
        mat1.add(val)
        
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat1.to_numpy()))
    
    def test_matrix_sub_matrix(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = np.random.randn(m, n).astype(np.float32)
        arr3 = arr1 - arr2
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = Matrix.matrix_from_array(arr2)
        mat3 = mat1.sub(mat2, inplace=False)
        mat1.sub(mat2)
        
        self.assertTrue(np.array_equal(arr3, mat3.to_numpy()))
        self.assertTrue(np.array_equal(arr3, mat1.to_numpy()))
    
    def test_matrix_sub_scalar(self):
        m, n = np.random.randint(100, 1000, (2,))
        val = np.random.randn()
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1 - val
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.sub(val, inplace=False)
        mat1.sub(val)
        
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat1.to_numpy()))
    
    def test_matrix_mul_matrix(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = np.random.randn(m, n).astype(np.float32)
        arr3 = arr1 * arr2
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = Matrix.matrix_from_array(arr2)
        mat3 = mat1.mul(mat2, inplace=False)
        mat1.mul(mat2)
        
        self.assertTrue(np.array_equal(arr3, mat3.to_numpy()))
        self.assertTrue(np.array_equal(arr3, mat1.to_numpy()))
    
    def test_matrix_mul_scalar(self):
        m, n = np.random.randint(100, 1000, (2,))
        val = np.random.randn()
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1 * val
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.mul(val, inplace=False)
        mat1.mul(val)
        
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat1.to_numpy()))
    
    def test_matrix_div_matrix(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = np.random.randn(m, n).astype(np.float32)
        arr3 = arr1 / arr2
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = Matrix.matrix_from_array(arr2)
        mat3 = mat1.div(mat2, inplace=False)
        mat1.div(mat2)
        
        self.assertTrue(np.array_equal(arr3, mat3.to_numpy()))
        self.assertTrue(np.array_equal(arr3, mat1.to_numpy()))
    
    def test_matrix_div_scalar(self):
        m, n = np.random.randint(100, 1000, (2,))
        val = np.random.randn()
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1 / val
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.div(val, inplace=False)
        mat1.div(val)
        
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat1.to_numpy()))
    
    def test_get_and_set_cols(self):
        m, n = 100, 50
        start, end = 12, 34
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1[:, start:end].copy()
        arr3 = np.random.randn(m, end - start).astype(np.float32)
        arr4 = arr1.copy()
        arr4[:, start:end] = arr3
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.get_cols(start, end, view=False)
        mat3 = Matrix.matrix_from_array(arr3)
        mat1.set_cols(start, end, mat3)
        
        self.assertTrue(np.array_equal(arr4, mat1.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        
    def test_get_and_set_rows(self):
        m, n = 100, 500
        start, end = 12, 98
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1[start:end].copy()
        arr3 = np.random.randn(end - start, n).astype(np.float32)
        arr4 = arr1.copy()
        arr4[start:end, :] = arr3
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.get_rows(start, end)
        mat3 = Matrix.matrix_from_array(arr3)
        mat1.set_rows(start,  end, mat3)
        
        self.assertTrue(np.array_equal(arr4, mat1.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
    
    def test_transpose(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1.transpose()
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.transpose(inplace=False)
        mat1.transpose()

        self.assertTrue(np.array_equal(arr2, mat1.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
    
    def test_sum(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1.sum(axis=0, keepdims=True)
        arr3 = arr1.sum(axis=1, keepdims=True)
        result1 = arr1.sum()
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.sum(axis=0)
        mat3 = mat1.sum(axis=1)
        result2 = mat1.sum()
        
        self.assertTrue(np.all(np.isclose(arr2, mat2.to_numpy(), atol=1e-4)))
        self.assertTrue(np.all(np.isclose(arr3, mat3.to_numpy(), atol=1e-4)))
        self.assertTrue(np.isclose(result1, result2, atol=1e-4))
    
    def test_mean(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1.mean(axis=0, keepdims=True)
        arr3 = arr1.mean(axis=1, keepdims=True)
        result1 = arr1.mean()
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.mean(axis=0)
        mat3 = mat1.mean(axis=1)
        result2 = mat1.mean()
        
        self.assertTrue(np.all(np.isclose(arr2, mat2.to_numpy(), atol=1e-4)))
        self.assertTrue(np.all(np.isclose(arr3, mat3.to_numpy(), atol=1e-4)))
        self.assertTrue(np.isclose(result1, result2, atol=1e-4))
    
    def test_less_than_matrix(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = np.random.randn(m, n).astype(np.float32)
        arr3 = (arr1 < arr2) * np.ones((m ,n), dtype=np.float32)
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = Matrix.matrix_from_array(arr2)
        mat3 = mat1.less_than(mat2, inplace=False)
        mat4 = mat1.less_than(mat2)
        
        self.assertTrue(np.array_equal(arr3, mat3.to_numpy()))
        self.assertTrue(np.array_equal(arr3, mat4.to_numpy()))
    
    def test_less_than_scalar(self):
        m, n = np.random.randint(100, 1000, (2,))
        val = np.random.randn()
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = (arr1 < val) * np.ones((m ,n), dtype=np.float32)
        
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.less_than(val, inplace=False)
        mat3 = mat1.less_than(val)
        
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat3.to_numpy()))
    
    def test_greater_than_matrix(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = np.random.randn(m, n).astype(np.float32)
        arr3 = (arr1 > arr2) * np.ones((m ,n), dtype=np.float32)
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = Matrix.matrix_from_array(arr2)
        mat3 = mat1.greater_than(mat2, inplace=False)
        mat4 = mat1.greater_than(mat2)
        
        self.assertTrue(np.array_equal(arr3, mat3.to_numpy()))
        self.assertTrue(np.array_equal(arr3, mat4.to_numpy()))
    
    def test_greater_than_scalar(self):
        m, n = np.random.randint(100, 1000, (2,))
        val = np.random.randn()
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = (arr1 > val) * np.ones((m ,n), dtype=np.float32)
        
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.greater_than(val, inplace=False)
        mat3 = mat1.greater_than(val)
        
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat3.to_numpy()))
        
    def test_equal_to_matrix(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = np.random.randn(m, n).astype(np.float32)
        arr3 = (arr1 == arr2) * np.ones((m ,n), dtype=np.float32)
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = Matrix.matrix_from_array(arr2)
        mat3 = mat1.equal_to(mat2, inplace=False)
        mat4 = mat1.equal_to(mat2)
        
        self.assertTrue(np.array_equal(arr3, mat3.to_numpy()))
        self.assertTrue(np.array_equal(arr3, mat4.to_numpy()))
    
    def test_equal_to_scalar(self):
        m, n = np.random.randint(100, 1000, (2,))
        val = np.random.randn()
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = (arr1 == val) * np.ones((m ,n), dtype=np.float32)
        
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.equal_to(val, inplace=False)
        mat3 = mat1.equal_to(val)
        
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        self.assertTrue(np.array_equal(arr2, mat3.to_numpy()))
    
    def test_min(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1.min(axis=0, keepdims=True)
        arr3 = arr1.min(axis=1, keepdims=True)
        result1 = arr1.min()
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.min(axis=0)
        mat3 = mat1.min(axis=1)
        result2 = mat1.min()
        
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        self.assertTrue(np.array_equal(arr3, mat3.to_numpy()))
        self.assertTrue(np.isclose(result1, result2, atol=1e-4))
    
    def test_max(self):
        m, n = np.random.randint(100, 1000, (2,))
        arr1 = np.random.randn(m, n).astype(np.float32)
        arr2 = arr1.max(axis=0, keepdims=True)
        arr3 = arr1.max(axis=1, keepdims=True)
        result1 = arr1.max()
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = mat1.max(axis=0)
        mat3 = mat1.max(axis=1)
        result2 = mat1.max()
        
        self.assertTrue(np.array_equal(arr2, mat2.to_numpy()))
        self.assertTrue(np.array_equal(arr3, mat3.to_numpy()))
        self.assertTrue(np.isclose(result1, result2, atol=1e-4))
    
    def test_dot(self):
        m, k, n = np.random.randint(100, 1000, (3,))
        self._test_dot(m, k, n)
        self._test_dot(1, k, n)
        self._test_dot(m, 1, n)
        self._test_dot(m, k, 1)
    
    def _test_dot(self, m: int, k: int, n: int):
        arr1 = np.random.randn(m, k).astype(np.float32)
        arr2 = np.random.randn(k, n).astype(np.float32)
        arr3 = arr1 @ arr2
        
        mat1 = Matrix.matrix_from_array(arr1)
        mat2 = Matrix.matrix_from_array(arr2)
        mat3 = mat1.dot(mat2)
        
        self.assertTrue(np.all(np.isclose(arr3, mat3.to_numpy(), atol=1e-4)))
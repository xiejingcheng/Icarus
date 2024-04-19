from lib_load import LibLoader
import numpy as np

lib = LibLoader("/h3cstore_ns/jcxie/Icarus/temp/kernel/build/libica.so")

A_arr = np.random.rand(1024,1024).astype(np.float32)
B_arr = np.random.rand(1024,1024).astype(np.float32)

D_arr = np.random.rand(1024,1024).astype(np.float32)

print(A_arr)
print(B_arr)


A = lib.build_matrix_from_array(A_arr)
B = lib.build_matrix_from_array(B_arr)
D = lib.build_matrix_from_array(D_arr)


lib.copy_to_device(A)
lib.copy_to_device(B)
lib.copy_to_device(D)

C = lib.gemm_on_device(A, B)

C_arr = lib.from_matrix_to_array(C)

print(C_arr)

print(np.dot(A_arr, B_arr))


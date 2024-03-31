from lib_load import LibLoader

_lib = LibLoader()

# _lib.print_device()

A = _lib.build_matrix_empty(1024, 1024)
B = _lib.build_matrix_empty(1024, 1024)
# _lib.copy_to_device(A)
# _lib.copy_to_device(B)

for i in range(10000):
    C = _lib.gemm_on_device(A, B)
_lib.copy_to_host(C)
_lib.print_tensor(C)
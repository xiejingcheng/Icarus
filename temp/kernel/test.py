from lib_load import LibLoader

_lib = LibLoader("/h3cstore_ns/jcxie/Icarus/kernel/build/libica.so")

# _lib.print_device()

A = _lib.build_matrix_empty(16, 16)
B = _lib.build_matrix_empty(16, 16)
_lib.copy_to_device(A)
_lib.copy_to_device(B)

C = _lib.gemm_on_device(A, B)
_lib.copy_to_host(C)
_lib.print_tensor(C)
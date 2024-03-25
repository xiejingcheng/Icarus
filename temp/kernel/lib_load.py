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


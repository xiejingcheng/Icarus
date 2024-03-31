if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np 
from dezero import Variable
from dezero.utils import plotDotGraph
import dezero.functions as F

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sin(x)

y.backward(createGraph=True)
print(x.grad)
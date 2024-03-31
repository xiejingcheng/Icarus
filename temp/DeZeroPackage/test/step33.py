if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np 
from dezero import Variable
from dezero.utils import plotDotGraph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(createGraph=True)
plotDotGraph(y, verbose=False, to_file='tanh.png')

iters = 5

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(createGraph=True)

gx = x.grad
gx.name = 'gx' + str(iters + 1)
plotDotGraph(gx, verbose=False, to_file='tanhGx.png')
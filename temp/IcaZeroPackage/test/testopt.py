if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from icazero import Variable
from icazero import optimizers
import icazero.functions as F
from icazero.models import MLP

x = np.random.randn(100, 1)
y = np.sin(2 * np.pi * x) + np.random.randn(100, 1)

lr = 0.2
max_iter = 1000000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.Adam(lr)
optimizer.setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.meanSquaredError(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(loss)
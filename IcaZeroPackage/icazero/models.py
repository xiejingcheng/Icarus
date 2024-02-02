from icazero import layers, Layer
from icazero import utils
import icazero.functions as F
import numpy as np

class Model(Layer):
    def plot(self, *inputs, toFile='model.png'):
        y = self.forward(*inputs)
        return utils.plotDotGraph(y, verbose=True, toFile=toFile)
    
    def save(self, path):
        self.to_cpu()
        params = [p.data for p in self.params()]
        np.savez_compressed(path, *params)
    
    def load(self, path):
        npz = np.load(path)
        params = [p.data for p in self.params()]
        for p, param in zip(self.params(), params):
            p.data = param

class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []
        
        for i, out_size in enumerate(fc_output_sizes):
            layer = layers.Linear(out_size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
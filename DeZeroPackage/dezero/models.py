from dezero import Layer
from dezero import utils
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
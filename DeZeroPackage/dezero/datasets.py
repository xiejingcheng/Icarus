import numpy as np 

class Dataset:
    def __init__(self, train=True, transform=None, targetTransform=None):
        self.train = train 
        self.transform = transform
        self.targetTransform = targetTransform

        if self.transform is None:
            self.transform = lambda x: x
        if self.targetTransform is None:
            self.targetTransform = lambda x: x

        self.data = None 
        self.label = None 
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.targetTransform(self.label[index])
        
    def __len__(self):
        return len(self.data)
    
    def prepare(self):
        pass


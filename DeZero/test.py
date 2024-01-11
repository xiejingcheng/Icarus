import numpy as np
from function import Square, Exp, square, exp
from core import Variable, Function



x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

# print(y.creator)
# assert y.creator == C
# assert y.creator.input is b
# assert y.creator.input.creator is B
# assert y.creator.input.creator.input is a
# assert y.creator.input.creator.input.creator is A
# assert y.creator.input.creator.input.creator.input is x

print(y.data)

y.grad = np.array(1.0)
y.backward()
print(x.grad)




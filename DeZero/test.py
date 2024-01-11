import numpy as np
from function import Square, Exp, square, exp, add
from core import Variable, Function



x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))


# print(y.creator)
# assert y.creator == C
# assert y.creator.input is b
# assert y.creator.input.creator is B
# assert y.creator.input.creator.input is a
# assert y.creator.input.creator.input.creator is A
# assert y.creator.input.creator.input.creator.input is x

y.backward()
print(y.data)
print(x.grad)




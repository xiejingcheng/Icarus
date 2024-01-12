is_simple_core = False

if is_simple_core:

    from dezero.simple_core import Variable
    from dezero.simple_core import Function
    from dezero.simple_core import usingConfig
    from dezero.simple_core import noGrad
    from dezero.simple_core import asVariable
    from dezero.simple_core import asArray
    from dezero.simple_core import setupVariable

else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import usingConfig
    from dezero.core import noGrad
    from dezero.core import asVariable
    from dezero.core import asArray
    from dezero.core import setupVariable

setupVariable()

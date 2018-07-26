import numpy as np

def probability(x,xj,dG):
    def num(xj,dG):
        out1 = xj*dG(xj)
        if out1 < 0:
            raise ValueError('The numerator is negative')
        return out1

    def denom(x,dG):
        out2 = np.sum(np.multiply(x,[dG(i) for i in x]))
        if out2 <= 0:
            raise ValueError('The denominator sucks')
        return out2

    return num(xj, dG) / denom(x, dG)

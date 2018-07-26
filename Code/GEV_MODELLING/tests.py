import numpy as np
from math import isclose

from GEV_MODELLING import data
from GEV_MODELLING import probability
from GEV_MODELLING.G import G
from GEV_MODELLING.G import dG




def test_probability_returns():

    d = data.data(100, 3, 3, [1,2,3,4], False)


    d.set_Xmat()
    d.generate_id_index()
    d.set_V()
    d.set_U()
    d.set_x()

    res = []
    for id in range(100):
        C = 0
        vec = d.xvec[d.IDvec[:,0] == id]
        for choice in vec:
            C2 = probability.probability(vec, choice, dG.logit)
            if C2 > 1 or C2 < 0:
                raise ValueError('Not all individual probabilities are 0')
            C = C + C2
        res.append(C)

    for sumprob in res[1:]:
        if not isclose(sumprob, 1, abs_tol = 0.0001):
            raise ValueError('Probabilities doesnt sum to one')

    print('No errors')

test_probability_returns()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns








from GEV_MODELLING.data import data, nesteddata

from GEV_MODELLING.utils import gather
from GEV_MODELLING import probability
from GEV_MODELLING.G import G
from GEV_MODELLING.G import dG
from GEV_MODELLING.likelihood import logli

from scipy.optimize import minimize

# This initializes data which are generated according to the model in data.py

def Beta(betalist, choices, xes):
    """ define a beta of proper size """
    return np.array(np.array(betalist).reshape(choices,xes))

true = 6
BETA = [true,1,2,5]

d = data(individuals = 10000, beta = Beta(BETA, 2, 2), run_on_init = True, include_cons = False)



l = logli(d)
loglik = [-l.logit([i,1,2,5,3,1], 3,2) for i in np.linspace(true-3, true +3, 100) ]
minim = minimize(lambda x: -l.logit(x, 3,2), [3,1,2,5,3,1])
plt.plot(np.linspace(true-3, true +3, 100), loglik)
plt.axvline(x=minim.x[0], color = 'b')
plt.axvline(x=true, color = 'black')
plt.show()


# THIS IS THE NESTED DATASET ===================================================
# === FIRST LETS CREATE A BASE DATASET =========================================
BETA1 = [1,3,3,2]
BETA2 = [-2,2,1,-1]

dgp = nesteddata((100, Beta(BETA1, 2, 2), dG.logit, True, False))
dgp.add_layer((100, Beta(BETA2, 2, 2), dG.logit, True, False))


df = gather(dgp.data, 'nest','choice', ['C_x', 'C_y'])

df

df['node'] = df['nest'] + df['choice'].astype(int).astype(str)



nodes = set('R').union(set(df['node']))
nest = set(df['nest'])


# === WE NEED TO DECIDE WHICH ROWS ARE VALID IN OUR TREE STRUCTURE =============

l = [('C_x', 0, 1),
('C_x', 1, 0),
('C_y', 0, 0),
('C_y', 1, 0)]

dat = {'nest':[], 'choice':[], 'alpha':[]}
for i in l:
    dat['nest'].append(i[0])
    dat['choice'].append(i[1])
    dat['alpha'].append(i[2])


df = df.merge(pd.DataFrame(dat), on = ['choice', 'nest'])


df


# CHECK PLOT TO TEST DISTRIBUTION OF DATA
sns.kdeplot(dgp.data.groupby('ID').mean().X1_x, dgp.data.groupby('ID').mean().X2_x, shade = True, color = 'b')
sns.kdeplot(dgp.data.groupby('ID').mean().X1_y, dgp.data.groupby('ID').mean().X2_y)
plt.show()

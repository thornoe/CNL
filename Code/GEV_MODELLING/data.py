import numpy as np
from GEV_MODELLING import probability
from GEV_MODELLING.G import G
from GEV_MODELLING.G import dG

from math import ceil
import pandas as pd
from functools import reduce


class data:
    """ Generates multinomial data """
    def __init__(self, individuals,  beta, dG = dG.logit, run_on_init = True, include_cons = False):

        self.beta = beta
        self.betashape = beta.shape
        self.individuals = individuals
        self.choices = self.betashape[0] #choices
        self.rows = self.individuals*self.choices
        self.cols = self.betashape[1] #cols

        self.G = G
        self.dG = dG

        self.has_cons = include_cons
        # if self.has_cons == True:
        #     if not self.cols + 1 == :
        #         raise ValueError('X and beta must have matching dimensions')
        # else:
        #     if not self.cols == len(beta):
        #         raise ValueError('X and beta must have matching dimensions')

        self.Xmat = None            # X variables and a col of 1's
        self.Vvec = None            # Utilities without noise
        self.xvec = None            # exp(utilities)
        self.xvec_noisy = None      # exp(utilities + noise)
        self.Uvec = None            # Noisy utilities
        self.IDmat = None           # vector of id-choice indices
        self.yvec = None            # Actual observed choice

        self.Cvec = None            # Choice options
        self.IDvec = None           # Individual ID's

        self.Pcol = None            # Column of Choice Proabilities
        self.Ccol = None            # Column of choice dummies

        if run_on_init:
            self.set_Xmat()
            self.generate_id_index()
            self.set_V()
            self.set_U()
            self.set_x()
            self.set_x_noisy()
            self.set_y()

    def set_Xmat(self, inherits = False):
        """ Generate a dataset of size (rows, cols)
        """
        if inherits == False:
            X_choices = np.random.normal(size = (self.individuals, self.cols))
            X = X_choices.repeat(self.choices, axis = 0)

        else:
            X_choices = np.random.normal(self.individuals - inherits.shape[1], self.cols)
            X = np.c_[inherits, x_choices]

        if self.has_cons:
            cons = np.ones(shape = (self.rows,1))
            x = np.c_[cons, X]
        else:
            x = X

        self.Xmat = x


    def set_V(self):
        """ Generate linear deterministic parts
        """
        if self.Xmat is None:
            raise ValueError('You need to set the X matrix')

        # i have a shitty day, so i guess im justified in writing shitty code ...
        n = 0

        while n < self.individuals:         # loop over individuals
            xn = self.Xmat[self.IDmat[:,0] == n + 1]   # This is the subset of X related to individual n
            IDn = self.IDmat[self.IDmat[:,0] == n + 1]

            for i in range(self.choices):
                if i == 0:
                    Vn = np.dot(xn[IDn[:,1] == i], self.beta[i])
                else:
                    Vn = np.vstack( (Vn,np.dot(xn[IDn[:,1] == i], self.beta[i]) ) )

            if n == 0:
                V = Vn
            else:
                V = np.vstack((V, Vn))
            n += 1
#        V = np.dot(self.Xmat, self.beta)
        self.Vvec = V


    def generate_id_index(self):
        """ produces indices for the dataset """
        IDvec = np.array(range(self.individuals))
        Cvec = np.array(range(self.choices))
        ID_C_vec = np.zeros((self.rows, 2))
        # this loops so that for each ID we have one observation per possible choice
        for _ in range(self.rows):
            ID_C_vec[_] = np.array([ceil((1 + _)/self.choices), (1 + _) % self.choices])

        self.IDmat = ID_C_vec

        self.Cvec = Cvec
        self.IDvec = IDvec


    def set_U(self):
        """ Adds noise to the V vector """
        if self.Vvec is None:
            raise ValueError('No V set')

        self.Uvec = self.Vvec + np.random.gumbel(0,1, len(self.Vvec))

    def set_x(self):
        """ The x in the G function
        """
        if self.Vvec is None:
            raise ValueError('You must set the V vector')
        self.xvec = np.exp(self.Vvec)


    def set_x_noisy(self):
        """ The x in the G function (with gumbel noise)
        """
        if self.Uvec is None:
            raise ValueError('You must set the U vector')
        self.xvec_noisy = np.exp(self.Uvec)


    def set_y(self):
        """ Uses probability to draw an outcome in the choice set """

        Pcol = []               # Column that will hold probabilities of each choice for each ID
        Crow = []               # Actual realized choice
        for i in self.IDvec:                # For each individual
            P = []                                  # Initialize an empty list
            C = self.Cvec                           # And a list with each choice
            idrows = np.c_[self.IDmat, self.xvec][np.c_[self.IDmat, self.xvec][:,0] == i + 1]           # Now find the rows of x'es where
                                                                                                        # we store this specific individual
            for c in self.Cvec:                                                                             # And loop over the choice set
                P.append(probability.probability(idrows[:,2], idrows[c,2], self.dG))                                # Calculating the prob
                                                                                                                    # of each choice as we go
            choice = np.random.choice(self.Cvec, p=P)                                                       # Then as we have the probability
                                                                                                            # of individual i choosing c forall c
            Crow.append([1 if x == choice else 0 for x in C])                                               # Calculate the actual choice (nondet)
            Pcol.append(P)

        self.Pcol = np.array(Pcol).flatten()
        self.Ccol = np.array(Crow).flatten()




class nesteddata(data):
    """ Extends the data class to generate nested datasets
    """
    def __init__(self, baseopts):
        """ Takes data() as base dataset, on which we build a nested dataset """
        super().__init__(*baseopts)

        self.data = pd.DataFrame(np.c_[self.IDmat, self.Xmat, self.Pcol, self.Ccol], columns = ['ID','option', 'X1', 'X2', 'P', 'C'])
        self.alpha = None
        # this enables easy merging when adding layers
        self.nests = [self.data]

    def add_layer(self, baseopts, inheritted = False):
        """ Adds a choice layer """
        layer = data(*baseopts)
        self.nests.append(layer)

        layer_data = pd.DataFrame(np.c_[layer.IDmat, layer.Xmat, layer.Pcol, layer.Ccol], columns = ['ID','option', 'X1', 'X2', 'P', 'C'])

        new_data = reduce(lambda x, y: pd.merge(x, y, on = 'ID'), [self.data, layer_data])
        self.data = new_data

    # @staticmethod
    # def _gather( df, key, value, cols ):
    #     id_vars = [ col for col in df.columns if col not in cols ]
    #     id_values = cols
    #     var_name = key
    #     value_name = value
    #     return pd.melt( df, id_vars, id_values, var_name, value_name )
    #
    #
    # def prepare_data(self, nest_list):
    #     self.df = self._gather(self.data, 'nest', 'choices', nest_list)

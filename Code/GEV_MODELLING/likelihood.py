import numpy as np
import time

class logli:
    def __init__(self, data):
        self.data = data
        self.X = self.data.Xmat


    def _Beta(self, betalist, choices, xes):
        return np.array(np.array(betalist).reshape(choices,xes))


    def generalized(self):
        out = np.sum(np.multiply(self.data.Ccol, np.log(self.data.Pcol)))
        return out

    def logit(self, beta_guess_list, c, x):
        """ Calculates values from the likelihood function of the logit model
        """

        final = 0
        n = 0
        beta_guess = self._Beta(beta_guess_list, c, x)

        while n < self.data.individuals:         # loop over individuals
            xn = self.X[self.data.IDmat[:,0] == n + 1]   # This is the subset of X related to individual n

            j = 0
            while j < self.data.choices:       # loop over choices
                # ==============================================================
                if self.data.Ccol[self.data.IDmat[:,0] == n + 1][j] == 1:      # if this is the choice made

                    xb = np.dot(xn[j], beta_guess[j])

                    # ==========================================================
                    # This segment calculated the last part of the log lik func
                    m = 0
                    partsum = 0
                    while m < self.data.choices:
                         partsum += np.exp(np.dot(xn[m], beta_guess[m]))
                         m += 1

                    inner = xb - np.log(partsum)

                    # ==========================================================
                else:
                    inner = 0
                # ==============================================================
                final += inner

                j += 1
            n += 1

        return final


        def CNL():
            pass

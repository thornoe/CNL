import numpy as np

class G:
    @staticmethod
    def logit(x, mu = 1):
        """ Logit GEV characterization """
        return np.sum(x)**mu


    @staticmethod
    def nested_logit(x, alpha):
        """NL GEV characterization """
        if utils.nests_are_crossed(alpha):
            raise ValueError('Nests are crossed!')




class dG:
    def __init__(self):
        pass

    @staticmethod
    def logit(x, opts = None, mu = 1):
        """Logit GEV char derivative """
        if mu == 1:
            return 1
        else:
            return mu*np.sum(x)**(mu - 1)



    @staticmethod
    def nested_logit(x, alpha):
        pass


    @staticmethod
    # {alpha, i, , mu_m, nests, choices}
    def CNL(x, opts = None, mu = 1):

        outer = 0

        for m in range(opts[nests]):
            inner = 0

            for j in range(opts[choices]):

                inner += opts[alpha][j][m]*x[j]**(opts[mu_m][m])

            out = opts[alpha][i][m]*x[i]**(opts[mu_m][m] - 1)
            outer += outer + inner**((opts[mu_m][m]/m) - 1)

        return mu*outer



class utils:
    """ Class of more or less random utility functions """
    @staticmethod
    def nests_are_crossed(alpha):
        """ Check if a nesting structure is crossed """
        inputset = set(alpha.flatten())

        if inputset == {0,1}:
            return False
        else:
            return True

from types import SimpleNamespace
import numpy as np

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        
        # total endowment
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):
        ## Imposing restriction that demand can't be negative (utility would be a complex number, I think)
        x1A=np.maximum(x1A, 0)
        x2A=np.maximum(x2A, 0)

        alpha = self.par.alpha
        util = x1A**alpha * x2A**(1 - alpha)
        return util

    def utility_B(self,x1B,x2B):

        ## Imposing restriction that demand can't be negative (utility would be a complex number, I think)
        x1B=np.maximum(x1B, 0)
        x2B=np.maximum(x2B, 0)

        beta = self.par.beta
        util = x1B**beta * x2B**(1 - beta)
        return util

    def demand_A(self,p1):
        w1A, w2A = self.par.w1A, self.par.w2A
        alpha = self.par.alpha
        x1A = alpha * (w1A * p1 + w2A) / p1
        x2A = (1 - alpha) * (w1A * p1 + w2A)
        return x1A, x2A

    def demand_B(self,p1):
        w1B, w2B = (1 - self.par.w1A), (1 - self.par.w2A)
        beta = self.par.beta
        x1B = beta * (w1B * p1 + w2B) / p1
        x2B = (1 - beta) * (w1B * p1 + w2B)
        return x1B, x2B

    def market_clear_err(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-par.w1B
        eps2 = x2A-par.w2A + x2B-par.w2B

        return eps1,eps2
    
    def find_eq(self, P_1):

        eps_1,eps_2=self.market_clear_err(P_1)

        # Use python-built in functions to find the minimum absolute error
        min_err1=abs(eps_1).min()
        min_err2=abs(eps_2).min()

        # Make vector indices to pass to price vector (make sure they are the same)
        ids1=abs(eps_1)==min_err1
        ids2=abs(eps_2)==min_err2
        assert ids1[0] == ids2[0]

        # Calculate market clearing price of price in P_1
        market_clearing_p=P_1[ids1][0]

        return market_clearing_p
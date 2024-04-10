import numpy as np
from types import SimpleNamespace



class ExchangeEconomyClass:
    '''A class of two agents in an exchange economy that maximizes utililty based on (initial) endowments and preference parameters.
    
    Attributes:
        utility_A/utility_B: Utility of agents A&B.
        demand_A/demand_B: Demands of agents A&B.
        market_clear_price: Price that clears the market.
        market_clear_err: Computes the market clear error w.r.t. a price.
    '''

    def __init__(self, alpha: float=1/3, beta: float=2/3, w1A: float=0.8, w2A: float=0.3):
        '''
        Initialize exchange economy class.
        Change model parameters as needed to analyze solutions.

        Parameters:
            alpha (float): Preference parameter for agent A.
            beta (float): Preference parameter for agent B.
            w1A (float): A's endowment of good 1.
            w2A (float): A's endowment of good 2.
        '''

        par = self.par = SimpleNamespace()

        par.alpha = alpha
        par.beta = beta
        par.w1A = w1A
        par.w2A = w2A
        par.w1B = 1-w1A
        par.w2B = 1-w2A

    def utility_A(self,x1A,x2A):
        '''
        Calculates utility for agent A, based on his/her demand.

        Parameters:
            x1A (float/int): A's demand for good 1.
            x2A (float/int): A's demand for good 2.
        
        Returns:
        float/int: Numeric utility value.
        '''

        par = self.par

        alpha = par.alpha

        util = x1A**alpha * x2A**(1 - alpha)
        return util

    def utility_B(self,x1B,x2B):
        '''
        Calculates utility for agent B, based on his/her demand.

        Parameters:
            x1B (float/int): B's demand for good 1.
            x2B (float/int): B's demand for good 2.
        
        Returns:
        float/int: Numeric utility value.
        '''

        par = self.par

        beta = par.beta
        
        util = x1B**beta * x2B**(1 - beta)

        return util

    def demand_A(self,p1):
        '''
        Computes A's demand based on a price with his/her preferences inherited from the model parameters.

        Parameters:
            p1 (float/int): Price that agent B optimizes his/her demands on.
        Returns:
            x1A, X2A (tuple, float/int): Tuple of quantity demanded by agent B
        '''
        

        par = self.par

        w1A, w2A = par.w1A, par.w2A
        alpha = par.alpha
        x1A = alpha * (w1A * p1 + w2A) / p1
        x2A = (1 - alpha) * (w1A * p1 + w2A)

        # Ensuring Walras law applies
        x1A = np.clip(x1A,0,1)
        x2A = np.clip(x2A,0,1)

        return x1A, x2A

    def demand_B(self,p1):
        '''
        Computes B's demand based on a price with his/her preferences inherited from the model parameters.

        Parameters:
            p1 (float/int): Price that agent B optimizes his/her demands on.
        Returns:
            x1B, X2B (tuple, float/int): Tuple of quantity demanded by agent B
        '''
        par = self.par

        w1B, w2B = (1 - par.w1A), (1 - par.w2A)
        beta = par.beta
        x1B = beta * (w1B * p1 + w2B) / p1
        x2B = (1 - beta) * (w1B * p1 + w2B)

        # Ensuring Walras law applies
        x1B = np.clip(x1B,0,1)
        x2B = np.clip(x2B,0,1)

        return x1B, x2B

    def market_clear_err(self,p1):
        '''
        Calculates the error in market clearing when based on demands that clears the market.

        Args:
            p1 (float/int): The price used in the market.
        Returns:
            eps1, eps2 (tuple(float)): Tuple of market errors
        '''
        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def market_clear_price(self, P_1):
        '''
        Based on market clearing error. This calculates the price that clears the market, but with lowest market clearing error.

        Args:
            P_1 (np.ndarray): Vector of possible prices.
        Returns:
            p_1 (float/int): Returns scalar of price that clears the market.
        '''
        eps_1,eps_2=self.market_clear_err(P_1)

        # Make a vector with difference between errors in markets
        EPS = abs(eps_1)+abs(eps_2)

        # Use python-built in functions to find the minimum absolute error
        minerr=EPS.min()

        # Make vector indices to pass to price vector
        ids=EPS==minerr

        # Calculate market clearing price of price in P_1
        market_clearing_p=P_1[ids][0]

        return market_clearing_p
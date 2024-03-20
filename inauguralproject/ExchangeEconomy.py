import numpy as np

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
            w1B (float): B's endowment of good 1 (1-w1A).
            w2B (float): B's endowment of good 2 (1-w2A).
        '''
        self.alpha = alpha
        self.beta = beta
        self.w1A = w1A
        self.w2A = w2A
        self.w1B = 1 - w1A
        self.w2B = 1 - w2A

    def utility_A(self,x1A,x2A):
        '''
        Calculates utility for agent A, based on his/her demand.

        Parameters:
            x1A (float/int): A's demand for good 1.
            x2A (float/int): A's demand for good 2.
        
        Returns:
        float/int: Numeric utility value.
        '''

        alpha = self.alpha

        ## Imposing restriction that demand can't be negative (utility would be a complex number, I think) or over 1
        x1A=np.clip(x1A, 0,1)
        x2A=np.clip(x2A, 0,1)

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
        
        beta = self.beta

        ## Imposing restriction that demand can't be negative (utility would be a complex number, I think) or over 1
        x1B=np.clip(x1B, 0,1)
        x2B=np.clip(x2B, 0,1)

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
        
        w1A, w2A = self.w1A, self.w2A
        alpha = self.alpha
        x1A = alpha * (w1A * p1 + w2A) / p1
        x2A = (1 - alpha) * (w1A * p1 + w2A)

        ## Imposing restriction that demand can't be negative (utility would be a complex number, I think) or over 1
        x1A=np.clip(x1A, 0,1)
        x2A=np.clip(x2A, 0,1)

        return x1A, x2A

    def demand_B(self,p1):
        '''
        Computes B's demand based on a price with his/her preferences inherited from the model parameters.

        Parameters:
            p1 (float/int): Price that agent B optimizes his/her demands on.
        Returns:
            x1B, X2B (tuple, float/int): Tuple of quantity demanded by agent B
        '''
        
        w1B, w2B = (1 - self.w1A), (1 - self.w2A)
        beta = self.beta
        x1B = beta * (w1B * p1 + w2B) / p1
        x2B = (1 - beta) * (w1B * p1 + w2B)

        ## Imposing restriction that demand can't be negative (utility would be a complex number, I think) or over 1
        x1B=np.clip(x1B, 0,1)
        x2B=np.clip(x2B, 0,1)
        return x1B, x2B

    def market_clear_err(self,p1):
        '''
        Calculates the error in market clearing when based on demands that clears the market.

        Args:
            p1: float
                The price used in the market.
        '''

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-self.w1A + x1B-self.w1B
        eps2 = x2A-self.w2A + x2B-self.w2B

        return eps1,eps2
    
    def market_clear_price(self, P_1):
        '''
        Based on market clearing error. This calculates the price that clears the market, but with lowest market clearing error.

        Args:
            P_1: np.ndarray
                Vector of possible prices.
        '''
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
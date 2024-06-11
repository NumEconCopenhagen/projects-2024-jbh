from scipy import optimize, interpolate
import numpy as np
import copy

class ConSavModel:
    def __init__(self, r: float = 0.05, beta: float = 0.95, rho: float | int = 5, gamma: float | int = None, delta: float | int = None, kappa: float | int = None, sigma_low: float | int = None, sigma_high: float | int = None, p: float = 0.5):

        '''
        Initialize exchange consumption-saving model class.
        Change model parameters as needed to analyze solutions.

        Parameters:
        -----

            r: float | int 
                Exogenous interest rate.
            beta: float | int
                Preference parameter for agent.
            rho: float | int
                Coefficient of risk aversion.
            gamma: float | int 
                Strength of bequest motive.
            delta: float | int
                "Risk" parameter.
            kappa: float | int
                Bequest parameter.
            sigma_low: float | int
                Under stochastic risk, lower boundary for risk value.
            sigma_high: float | int
                Under stochastic risk, upper boundary for risk value. 
            p: float | int
                Probability of ending in "low" income case.
        '''

        self.rho = rho
        self.r = r
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.kappa = kappa
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.p = p  

    def copy(self):
        '''
        Creates a copy of the current ConSavModel with same parameters.

        Returns:
        --------
            ConSavModel: 
                New instance of ConSavModel.
        '''
        return copy.deepcopy(self)

    def solve_cons_crra(self, m1):
        '''
        Solve for the optimal consumption in period 1 and 2 given intial endowments. Assumes a constant relative risk aversion utility function.

        Parameters:
        ------
            m1: float
                Intial wealth in period 1.

        Returns:
        --------
            tuple: 
                A tuple containing the optimal consumption in period 1 and 2.
        '''
        
        # Objective func
        obj = lambda c1: c1 - ((1+self.r)*m1)/((self.beta*(1+self.r))**(1/self.rho)+1+self.r)

        guess = m1/2

        # Solver
        res = optimize.root_scalar(obj, x0=guess, method='newton')
        c1_star = res.root

        # Based on c1, we can calculate c2
        c2_star = (1+self.r)*(m1-c1_star)

        return c1_star, c2_star

    def utility_crra(self, c):
        '''
        Constant relative risk aversion utility function.

        Parameters:
        ------
            c: float
                Consumption.

        Returns:
        --------
            float: 
                A numeric utility value.
        '''
        util = (c**(1-self.rho))/(1-self.rho)
        return util

    def v1_func(self, c1, m1, v2_interp_func):
        '''
        A (utility) value function in period 1. Assumes CRRA utility with bequest component. Includes expected utility in period 2.

        Parameters:
        ------
            c1: float | int
                Consumption in period 1.
            m1: float | int
                Wealth/endowment in period 1.

        Returns:
        --------
            float | int: 
                Value of utility in period 1 and expected utility in period 2.
        '''

        y2_low = 1-self.delta
        y2_high = 1+self.delta

        m2_low = (1+self.r)*(m1-c1)+y2_low
        m2_high = (1+self.r)*(m1-c1)+y2_high

        exp_v2_low = v2_interp_func([m2_low])[0]
        exp_v2_high = v2_interp_func([m2_high])[0]

        # total expected value in v2
        total = self.p * exp_v2_high + (1-self.p) * exp_v2_low

        return self.utility_crra(c1) + self.beta*total
    
    def v1_func_no_risk(self, c1, m1, v2_interp_func):
        '''
        A (utility) value function in period 1. Assumes CRRA utility with bequest component and no risk. Includes expected utility in period 2.

        Parameters:
        ------
            c1: float | int
                Consumption in period 1.
            m1: float | int
                Wealth/endowment in period 1.

        Returns:
        --------
            float | int: 
                Value of utility in period 1 and expected utility in period 2.
        '''

        y2=1
        m2 = (1+self.r)*(m1-c1)+y2

        exp_v2 = v2_interp_func([m2])[0]

        return self.utility_crra(c1) + self.beta*exp_v2


    def v1_func_stoch(self, c1, m1, sigma, v2_interp_func):
        '''
        A (utility) value function in period 1. Assumes CRRA utility with bequest component under stochastic income risk. Includes expected utility in period 2.

        Parameters:
        ------
            c1: float | int
                Consumption in period 1.
            m1: float | int
                Wealth/endowment in period 1.

        Returns:
        --------
            float | int: 
                Value of utility in period 1 and expected utility in period 2.
        '''
        m2_low = (1+self.r)*(m1-c1)*(1-sigma)
        m2_high = (1+self.r)*(m1-c1)*(1+sigma)

        exp_v2_low = v2_interp_func([m2_low])[0]
        exp_v2_high = v2_interp_func([m2_high])[0]

        # total expected value in v2
        total = self.p * exp_v2_high + (1-self.p)*exp_v2_low

        return self.utility_crra(c1) + self.beta*total

    def interp(self, m_grid, v_grid):
        '''
        A (linear) interpolate function to derive expected utility one period ahead of time. 

        Parameters:
        ------
            m_grid: np.ndarray
                Grid points from where we interpolate.
            v_grid: float | int
                Values of function being interpolated at each grid point.

        Returns:
        --------
            interpolator: 
                An interpolator object to interpolate at points within or outside grid.
        '''
        func = interpolate.RegularGridInterpolator((m_grid,), v_grid,
                                                        bounds_error=False,fill_value=None)
        return func
    
    def v2_func(self, c2, m2):
        '''
        A (utility) value function in period 2. Assumes CRRA utility with bequest component.

        Parameters:
        ------
            c2: float | int
                Consumption in period 2.
            m2: float
                Wealth in period 2.

        Returns:
        --------
            float | int: 
                Numeric value of utility in period 2.
        '''

        cons = (c2**(1-self.rho))/(1-self.rho)
        beq = self.gamma*(m2-c2+self.kappa)**(1-self.rho)/(1-self.rho)
        return cons+beq
    
    def solve_period_1(self, v2_interp_func, v1):
        '''
        Function that solves an agent's maximization problem in period 1.

        Parameters:
        ------
            v2_interp_func: interpolator object
                A function that interpolates values for period 2.
            v1: objective function
                The objective function of the agent in period 1.

        Returns:
        --------
            tuple: 
                A tuple of np.ndarrays with endowments (m1), value functions (v1) and consumption in period 1 (c1).
        '''

        m1_grid = np.linspace(1e-8,4,100)
        v1_grid = np.empty(100)
        c1_grid = np.empty(100)

        # For each m1 in grid
        for i,m1 in enumerate(m1_grid):
            
            # Defining obj func
            obj = lambda c1: -v1(c1, m1, v2_interp_func)
                    
            # Optimize bounded
            result = optimize.minimize_scalar(obj, method='bounded',bounds=(1e-8,m1))
            
            # Save results
            v1_grid[i] = -result.fun
            c1_grid[i] = result.x
        
        return m1_grid, v1_grid, c1_grid

    def solve_period_2(self):
        '''
        Function that solves an agent's maximization problem in period 2. Takes only previously defined (exogenous) parameters.

        Returns:
        --------
            tuple: 
                A tuple of np.ndarrays with endowments (m2), value functions (v2) and consumption in period 2 (c2).
        '''        
        # Defining grids
        m2_grid = np.linspace(1e-8,5,500)
        v2_grid = np.empty(500)
        c2_grid = np.empty(500)

        # b. solve for each m2 in grid
        for i, m2 in enumerate(m2_grid):
            
            # Defining obj func
            obj = lambda c2: -self.v2_func(c2, m2)

            # Optimize bounded
            result = optimize.minimize_scalar(obj, method='bounded', bounds=(1e-8,m2))
            
            # Save results
            v2_grid[i] = -result.fun
            c2_grid[i] = result.x
        
        return m2_grid, v2_grid, c2_grid

    def solve_period_1_no_risk(self, v2_interp_func, v1):
        '''
        Function that solves an agent's maximization problem in period 1 under no income risk.

        Parameters:
        ------
            v2_interp_func: interpolator object
                A function that interpolates values for period 2.
            v1: objective function
                The objective function of the agent in period 1.

        Returns:
        --------
            tuple: 
                A tuple of np.ndarrays with endowments (m1), value functions (v1) and consumption in period (c1).
        '''

        m1_grid = np.linspace(1e-8,4,300)
        v1_grid = np.empty(300)
        c1_grid = np.empty(300)

        # For each m1 in grid
        for i,m1 in enumerate(m1_grid):
            
            # Defining obj func
            obj = lambda c1: -v1(c1, m1, v2_interp_func)
                    
            # Optimize bounded
            result = optimize.minimize_scalar(obj, method='bounded',bounds=(1e-8,m1))
            
            # Save results
            v1_grid[i] = -result.fun
            c1_grid[i] = result.x
        
        return m1_grid,v1_grid,c1_grid
    
    def solve_period_1_stoch(self, v2_interp_func, v1):
        '''
        Function that solves an agent's maximization problem in period 1 under stochastic income risk. Risk parameters are defined when instantiating the model.

        Parameters:
        ------
            v2_interp_func: interpolator object
                A function that interpolates values for period 2.
            v1: objective function
                The objective function of the agent in period 1.

        Returns:
        --------
            tuple: 
                A tuple of np.ndarrays with endowments (m1), value functions (v1) and consumption in period (c1).
        '''
        m1_grid = np.linspace(1e-8,5,300)
        v1_grid = np.empty(300)
        c1_grid = np.empty(300)

        # For each m1 in grid
        for i,m1 in enumerate(m1_grid):
            
            sigma = np.random.uniform(self.sigma_low, self.sigma_high)

            # Defining obj func
            obj = lambda c1: -v1(c1, m1, sigma, v2_interp_func)
                    
            # Optimize bounded
            result = optimize.minimize_scalar(obj, method='bounded',bounds=(1e-8,m1))
            
            # Save results
            v1_grid[i] = -result.fun
            c1_grid[i] = result.x
        
        return m1_grid, v1_grid, c1_grid
    
    def solvez_no_risk(self, v1):
        '''
        Compound that solves the agent's maximization problem under no income risk in its entirety.

        Parameters:
        ------
            v1: objective function
                The objective function of the agent in period 1.

        Returns:
        --------
            tuple: 
                A tuple of np.ndarrays with endowments (m1/m2) and consumption in period 1 and period 2 (c1/c2).
        '''
        # Solving for period 2
        m2_grid, v2_grid, c2_grid = self.solve_period_2()

        # Interpolator
        v2_interp_func = self.interp(m2_grid, v2_grid)

        # Solving for period 1
        m1_grid, v1_grid, c1_grid = self.solve_period_1_no_risk(v2_interp_func, v1)
        
        return m1_grid, c1_grid, m2_grid, c2_grid

    def solvez(self, v1):
        '''
        Compound that solves the agent's maximization problem in its entirety.

        Parameters:
        ------
            v1: objective function
                The objective function of the agent in period 1.

        Returns:
        --------
            tuple: 
                A tuple of np.ndarrays with endowments (m1/m2) and consumption in period 1 and period 2 (c1/c2).
        '''        
        # Solving for period 2
        m2_grid, v2_grid, c2_grid = self.solve_period_2()

        # Interpolator
        v2_interp_func = self.interp(m2_grid, v2_grid)

        # Solving for period 1
        m1_grid, v1_grid, c1_grid = self.solve_period_1(v2_interp_func, v1)
        
        return m1_grid, c1_grid, m2_grid, c2_grid

    def solvez_stoch(self, v1):
        '''
        Compound that solves the agent's maximization problem under stochastic income risk in its entirety.

        Parameters:
        ------
            v1: objective function
                The objective function of the agent in period 1.

        Returns:
        --------
            tuple: 
                A tuple of np.ndarrays with endowments (m1/m2) and consumption in period 1 and period 2 (c1/c2).
        '''        
        # Solving for period 2
        m2_grid, v2_grid, c2_grid = self.solve_period_2()

        # Interpolator
        v2_interp_func = self.interp(m2_grid, v2_grid)

        # Solving for period 1
        m1_grid, v1_grid, c1_grid = self.solve_period_1_stoch(v2_interp_func, v1)
        
        return m1_grid, c1_grid, m2_grid, c2_grid
from scipy import optimize, interpolate
import numpy as np
import copy

class ConSavModel:
    def __init__(self, r: float = 0.05, beta: float = 0.95, rho: float | int = 5, gamma: float | int = None, delta: float | int = None, kappa: float | int = None, sigma_low: float | int = None, sigma_high: float | int = None, p: float = 0.5):

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
        return copy.deepcopy(self)

    def solve_cons_crra(self, m1):
        '''
        DESCRIPTION

        Parameters:
            - SOMETHING
        Returns:
            - SOMETHING ELSE
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

    def utility_crra(self, c1):
        util = (c1**(1-self.rho))/(1-self.rho)
        return util

    def v2_func(self, c2, m2):
        cons = (c2**(1-self.rho))/(1-self.rho)
        beq = self.gamma*(m2-c2+self.kappa)**(1-self.rho)/(1-self.rho)
        return cons+beq

    def v1_func(self, c1, m1, v2_interp_func):

        y2_low = 1-self.delta
        y2_high = 1+self.delta

        m2_low = (1+self.r)*(m1-c1)+y2_low
        m2_high = (1+self.r)*(m1-c1)+y2_high

        exp_v2_low = v2_interp_func([m2_low])[0]
        exp_v2_high = v2_interp_func([m2_high])[0]

        # total expected value in v2
        total = self.p * exp_v2_high + (1-self.p) * exp_v2_low

        return self.utility_crra(c1, self.rho) + self.beta*total
    
    def v1_func_no_risk(self, c1, m1, v2_interp_func):

        y2=1
        m2 = (1+self.r)*(m1-c1)+y2

        exp_v2 = v2_interp_func([m2])[0]

        return self.utility_crra(c1) + self.beta*exp_v2


    def v1_func_stoch(self, c1, m1, sigma, v2_interp_func):

        m2_low = (1+self.r)*(m1-c1)*(1-sigma)
        m2_high = (1+self.r)*(m1-c1)*(1+sigma)

        exp_v2_low = v2_interp_func([m2_low])[0]
        exp_v2_high = v2_interp_func([m2_high])[0]

        # total expected value in v2
        total = self.p * exp_v2_high + (1-self.p)*exp_v2_low

        return self.utility_crra(c1) + self.beta*total

    def interp(self, m_grid, v_grid):
        func = interpolate.RegularGridInterpolator((m_grid,), v_grid,
                                                        bounds_error=False,fill_value=None)
        return func

    def solve_period_1(self, v2_interp_func, v1):
        
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
        
        return m1_grid,v1_grid,c1_grid

    def solve_period_2(self):
        
        # Defining grids
        m2_grid = np.linspace(1e-8,5,500)
        v2_grid = np.empty(500)
        c2_grid = np.empty(500)

        # b. solve for each m2 in grid
        for i,m2 in enumerate(m2_grid):
            
            # Defining obj func
            obj = lambda c2: -self.v2_func(c2, m2)

            # Optimize bounded
            result = optimize.minimize_scalar(obj, method='bounded', bounds=(1e-8,m2))
            
            # Save results
            v2_grid[i] = -result.fun
            c2_grid[i] = result.x
        
        return m2_grid, v2_grid, c2_grid

    def solve_period_1_no_risk(self, v2_interp_func, v1):
        
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

    def solvez_no_risk(self, v1):
        
        # Solving for period 2
        m2_grid, v2_grid, c2_grid = self.solve_period_2()

        # Interpolator
        v2_interp_func = self.interp(m2_grid, v2_grid)

        # Solving for period 1
        m1_grid, v1_grid, c1_grid = self.solve_period_1_no_risk(v2_interp_func, v1)
        
        return m1_grid, c1_grid, m2_grid, c2_grid

    def solvez(self, v1):
        
        # Solving for period 2
        m2_grid, v2_grid, c2_grid = self.solve_period_2()

        # Interpolator
        v2_interp_func = self.interp(m2_grid, v2_grid)

        # Solving for period 1
        m1_grid, v1_grid, c1_grid = self.solve_period_1(v2_interp_func, v1)
        
        return m1_grid, c1_grid, m2_grid, c2_grid

    def solvez_stoch(self, v1):
        
        # Solving for period 2
        m2_grid, v2_grid, c2_grid = self.solve_period_2()

        # Interpolator
        v2_interp_func = self.interp(m2_grid, v2_grid)

        # Solving for period 1
        m1_grid, v1_grid, c1_grid = self.solve_period_1_stoch(v2_interp_func, v1)
        
        return m1_grid, c1_grid, m2_grid, c2_grid

    def solve_period_1_stoch(self, v2_interp_func, v1):
        
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
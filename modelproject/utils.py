from scipy import optimize, interpolate
import numpy as np

def solve_cons_crra(m1, r, beta, rho):
    
    # Objective func
    obj = lambda c1: c1 - ((1+r)*m1)/((beta*(1+r))**(1/rho)+1+r)

    guess = m1/2

    # Solver
    res = optimize.root_scalar(obj, x0=guess, method='newton')
    c1_star = res.root

    # Based on c1, we can calculate c2
    c2_star = (1+r)*(m1-c1_star)

    return c1_star, c2_star

# def solve_cons_log(m1, r, beta):
    
#     # Objective func
#     obj = lambda c1: (beta/(c1-m1))+1/c1

#     guess = m1/2

#     # Solver
#     res = optimize.root_scalar(obj, x0=guess, method='newton')
#     c1_star = res.root

#     # Based on c1, we can calculate c2
#     c2_star = (1+r)*(m1-c1_star)

#     return round(c1_star,3), round(c2_star,3)


def utility_crra(c1, rho):
    util = (c1**(1-rho))/(1-rho)
    return util

def v2_func(c2, rho, gamma, m2, kappa):
    cons = (c2**(1-rho))/(1-rho)
    beq = gamma*(m2-c2+kappa)**(1-rho)/(1-rho)
    return cons+beq

def v1_func(c1, m1, rho, beta, r, delta, v2_interp_func):

    y2_low = 1-delta
    y2_high = 1+delta

    m2_low = (1+r)*(m1-c1)+y2_low
    m2_high = (1+r)*(m1-c1)+y2_high

    exp_v2_low = v2_interp_func([m2_low])[0]
    exp_v2_high = v2_interp_func([m2_high])[0]

    # total expected value in v2
    total = 0.5 * exp_v2_low + 0.5 * exp_v2_high

    return utility_crra(c1, rho) + beta*total

def v1_func_stoch(c1, m1, rho, beta, r, delta, v2_interp_func):

    m2_low = (1+r)*(m1-c1)*(1-sigma)
    m2_high = (1+r)*(m1-c1)*(1+sigma)

    exp_v2_low = v2_interp_func([m2_low])[0]
    exp_v2_high = v2_interp_func([m2_high])[0]

    # total expected value in v2
    total = delta * exp_v2_low + (1-delta) * exp_v2_high

    return utility_crra(c1, rho) + beta*total

def interp(m_grid, v_grid):
    func = interpolate.RegularGridInterpolator((m_grid,), v_grid,
                                                    bounds_error=False,fill_value=None)
    return func

def solve_period_1(rho, beta, r, delta, v2_interp_func, v1):
    
    m1_grid = np.linspace(1e-8,4,100)
    v1_grid = np.empty(100)
    c1_grid = np.empty(100)

    # For each m1 in grid
    for i,m1 in enumerate(m1_grid):
        
        # Defining obj func
        obj = lambda c1: -v1(c1, m1, rho, beta, r, delta, v2_interp_func)
                
        # Optimize bounded
        result = optimize.minimize_scalar(obj, method='bounded',bounds=(1e-8,m1))
        
        # Save results
        v1_grid[i] = -result.fun
        c1_grid[i] = result.x
     
    return m1_grid,v1_grid,c1_grid

def solve_period_2(rho, kappa, gamma):
    
    # Defining grids
    m2_grid = np.linspace(1e-8,5,500)
    v2_grid = np.empty(500)
    c2_grid = np.empty(500)

    # b. solve for each m2 in grid
    for i,m2 in enumerate(m2_grid):
        
        # Defining obj func
        obj = lambda c2: -v2_func(c2, rho, gamma, m2, kappa)

        # Optimize bounded
        result = optimize.minimize_scalar(obj, method='bounded', bounds=(1e-8,m2))
        
        # Save results
        v2_grid[i] = -result.fun
        c2_grid[i] = result.x
     
    return m2_grid, v2_grid, c2_grid

def solvez(rho, kappa, gamma, beta, r, delta, v1):
    
    # Solving for period 2
    m2_grid, v2_grid, c2_grid = solve_period_2(rho, kappa, gamma)

    # Interpolator
    v2_interp_func = interp(m2_grid, v2_grid)

    # Solving for period 1
    m1_grid, v1_grid, c1_grid = solve_period_1(rho, beta, r, delta, v2_interp_func, v1)
    
    return m1_grid, c1_grid, m2_grid, c2_grid


def solve_period_1_stoch(rho, beta, r, delta, v2_interp_func, v1):
    
    m1_grid = np.linspace(1e-8,4,100)
    v1_grid = np.empty(100)
    c1_grid = np.empty(100)

    # For each m1 in grid
    for i,m1 in enumerate(m1_grid):
        
        sigma = np.random.uniform(0.01, 0.5)

        # Defining obj func
        obj = lambda c1: -v1(c1, m1, rho, beta, r, sigma, v2_interp_func)
                
        # Optimize bounded
        result = optimize.minimize_scalar(obj, method='bounded',bounds=(1e-8,m1))
        
        # Save results
        v1_grid[i] = -result.fun
        c1_grid[i] = result.x
     
    return m1_grid,v1_grid,c1_grid
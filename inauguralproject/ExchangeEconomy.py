from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

        # c. market clearing - Use the same naming convention
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):
        par = self.par
        return x1A**par.alpha * x2A**(1 - par.alpha)

    def utility_B(self,x1B,x2B):
        par = self.par
        return x1B**par.beta * x2B**(1 - par.beta)

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
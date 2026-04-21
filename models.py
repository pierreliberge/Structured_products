import numpy as np

class RateCurve:
    def __init__(self, maturities: list, rates: list, compounding):
        self.maturities = maturities
        self.rates = rates
        self.compounding = compounding

    def get_rate(self, t):
        if t <= self.maturities[0]:
            return self.rates[0]
        for i in range(0, len(self.maturities)-1):
            if t == self.maturities[i]:
                return self.rates[i]

            if self.maturities[i] < t  < self.maturities[i+1]:
                t1 = self.maturities[i]
                t2 = self.maturities[i+1]
                r1 = self.rates[i]
                r2 = self.rates[i+1]
                poids = (t - t1) / (t2 - t1)
                rate = r1 + poids * (r2 - r1)
                return rate

        if t == self.maturities[-1]:
            return self.rates[-1]




        raise ValueError("maturité non présente et pas d'interpolation possible")

    def discount_factor(self, t):
        rate = self.get_rate(t)
        if t == 0:
            return 1
        if self.compounding == "annual":
            DF = 1 / (1+rate)**t
            return DF
        elif self.compounding == "continuous":
            DF = np.exp(-rate*t)
            return DF



        raise ValueError("compounding non reconnu")


    def forward_rate(self, t1, t2):
        DF1 = self.discount_factor(t1)
        DF2 = self.discount_factor(t2)
        tau = t2 - t1
        if tau <= 0:
            raise ValueError("Problème de dates")
        forward = ((DF1 / DF2) - 1) / tau
        return forward



class BlackScholesModel:
    def __init__(self, sigma):
        if sigma <= 0:
            raise ValueError("Sigma doit être strictement positif")
        self.sigma = sigma



class GBMModel:
    def __init__(self, sigma):
        if sigma <= 0:
            raise ValueError("Sigma doit être strictement positif")
        self.sigma = sigma

    def simulate_paths(self, S0, r, T, q, n_paths, n_steps):
        if S0 <= 0:
            raise ValueError("Spot doit être strictement positif ")
        if n_paths <= 0:
            raise ValueError("Nombre de paths doit etre positif")
        if n_steps <= 0:
            raise ValueError("Nombre de steps doit etre positif")
        if T < 0:
            raise ValueError("Temps à maturité doit être positif")
        elif T == 0:
            matrix = np.ones((n_paths, n_steps + 1))
            price_matrix = matrix * S0
            return price_matrix
        else:
            matrix = np.ones((n_paths, n_steps+1))
            matrix[:, 0] = np.log(S0)
            dt = T / n_steps
            z = np.random.normal(0, 1, size=(n_paths, n_steps))
            increment = (r - q - (self.sigma**2/2)) * dt + (self.sigma * np.sqrt(dt) * z)
            increment = increment.cumsum(axis=1)
            matrix[:, 1:] = np.log(S0) + increment
            price_matrix = np.exp(matrix)
            return price_matrix

















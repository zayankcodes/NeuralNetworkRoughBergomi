from math import log, sqrt, exp, pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, integrate
from scipy.stats import norm, qmc
from scipy.integrate import quad, trapezoid, quad_vec
from scipy.optimize import brentq
from scipy.interpolate import SmoothBivariateSpline
from scipy.special import hyp2f1
from math import log, sqrt, exp

n=1000
m=100000

dw1 = np.random.randn(m, n).astype(np.float32)


def G(x, H):
    return (2 * H) / (1/2 + H) * x ** (1/2 - H) * hyp2f1(1, 1/2 - H, 3/2 + H, x)


def riemann_liouville(T, n, H, m):

    
    dt = T / n
    gamma = np.zeros((n, n), dtype=np.float32)

    G_vec = np.vectorize(G)

    j = np.arange(1, n+1, dtype=np.float32)   
    i = np.arange(1, n+1, dtype=np.float32)  
    J, I = np.meshgrid(j, i, indexing='ij')

    ratio = np.where(J <= I, J / I, I / J)
    scaling = np.where(J <= I, (J * dt) ** (2 * H), (I * dt) ** (2 * H))

    gamma = scaling * G_vec(ratio, H)

    L = np.linalg.cholesky(gamma)
    X = L @ dw1.T
    X = np.vstack((np.zeros((1, m)), X)).T

    return X


def variance(xi0, eta, riemann_liouville, T, H):
    
    time_grid = np.linspace(0,T,n)
    riemann_slice = riemann_liouville[:, :n]
    
    xi = (xi0[0] + xi0[1] * time_grid + xi0[2] * time_grid ** 2) * np.exp(eta * riemann_slice - 0.5 * (eta ** 2) * time_grid**(2*H))
    return xi


def dz(dw1, rho, n, m):

    dw2 = np.random.randn(m, n).astype(np.float32)
    return rho * dw1 + np.sqrt(1 - rho ** 2) * dw2



def mc_sim(S0, n, m, r, T, xi0, eta, rho, H, whole_process=False):
    riemann_liouville_x = riemann_liouville(T, n, H, m) 
    xi = variance(xi0, eta, riemann_liouville_x, T, H) 
    dz1 = dz(dw1, rho, n, m) 
    
    dt = T / n

    increments = ((r - 0.5 * xi) * dt + np.sqrt(xi) * np.sqrt(dt) * dz1).astype(np.float32)

    if whole_process:

        log_prices = np.log(S0) + np.cumsum(increments, axis=1)
        prices = np.exp(log_prices)
    else:
  
        log_prices = np.log(S0) + np.sum(increments, axis=1)
        prices = np.exp(log_prices)
    
    return prices




def rbergomi_price(S0, n, m, r, T, K, xi0, eta, rho, H):
    prices = mc_sim(S0, n, m, r, T, xi0, eta, rho, H)

    discount_factor = np.exp(-r * T)
    payoffs = np.maximum(prices - K, 0)

    return np.mean(payoffs) * discount_factor


def black_scholes_call_price(S0, K, r, T, sigma):
    if sigma <= 0:
        return max(S0 - K * np.exp(-r * T), 0.0)
    
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call




def implied_volatility(target_price, S0, K, r, T, tol=1e-8, max_iterations=100):
    objective = lambda sigma: black_scholes_call_price(S0, K, r, T, sigma) - target_price

    vol_lower = 1e-6
    vol_upper = 5.0  

    try:
        implied_vol = optimize.brentq(objective, vol_lower, vol_upper, xtol=tol, maxiter=max_iterations)
    except ValueError:
        implied_vol = np.nan

    return implied_vol 
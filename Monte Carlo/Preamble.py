import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# Calculate standard error of an array of payoffs
def standard_error(payoffs, r, T):
    payoff = np.mean(payoffs) * np.exp(-r * T)
    N = len(payoffs)
    sigma = np.sqrt(np.sum((payoffs - payoff)**2) / (N - 1))
    SE = sigma / np.sqrt(N)
    return SE

# Calculate the percentage of in-the-money paths
def per_in_money_paths(paths):
    N = len(paths)
    A = paths[paths > 0]
    M = len(A)
    return M / N

# Calculate implied volatility using bisection method
def implied_volatility(call_price, S, K, T, r, tol=1e-6):
    def objective_function(sigma):
        return black_scholes_option(S, K, T, r, sigma, 0, 'call') - call_price

    # Set lower and upper bounds for volatility
    sigma_low = 0.001
    sigma_high = 2.0

    # Use bisection method to find implied volatility
    implied_volatility = bisect(objective_function, sigma_low, sigma_high, xtol=tol)
    return implied_volatility

# Calculate option price using Black-Scholes formula
def black_scholes_option(S, K, T, r, sigma, q=0, option_type='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return option_price

# Generate geometric Brownian motion paths
def geo_paths(S, T, sigma, steps, N, r=0, q=0):
    dt = T / steps
    ST = S * np.exp(np.cumsum(((r - q - sigma**2 / 2) * dt +
                               sigma * np.sqrt(dt) *
                               np.random.normal(size=(steps, N))), axis=0))
    return ST

# Plot Monte Carlo simulation results with shaded areas representing standard errors
def plot_mc_sim(prices, color, label, r, T, title, a=0.5):
    price = [np.mean(x) * np.exp(-r * T) for x in prices]
    SE = [standard_error(x, r, T) for x in prices]
    
    for i in range(len(price)):
        p = price[i]
        se = SE[i]
        xx = np.linspace(p - 3 * se, p + 3 * se)
        ss = stats.norm.pdf(xx, p, se)
        c = color[i]
        l = label[i]
        
        plt.fill_between(xx, ss, color=c, alpha=a, label=l)
        plt.plot([p, p], [0, max(ss) * 1.1], 'k')
        plt.title(title, size=20)
    
    plt.legend()
    plt.show()

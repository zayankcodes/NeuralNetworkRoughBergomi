#  Rough Bergomi sample:(ξ0,ν,ρ,H) ∈U[0.01,0.16]8×U[0.5,4.0]×U[−0.95,−0.1]×U[0.025,0.5]


import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from rbergomi import mc_sim, implied_volatility, dw1

options_data = pd.read_csv('yfinance_dataset.csv')


n=1000
m=100000

r = 0.0427
S0 = 1

num_samples = 500
a_samples = np.random.uniform(0.002, 0.1, num_samples).astype(np.float32)
b_samples = np.random.uniform(0.002, 0.1, num_samples).astype(np.float32)
c_samples = np.random.uniform(0.002, 0.1, num_samples).astype(np.float32)
eta_samples = np.random.uniform(0.5, 4.0, num_samples).astype(np.float32)
rho_samples = np.random.uniform(-0.95, -0.1, num_samples).astype(np.float32)
H_samples = np.random.uniform(0.025, 0.5, num_samples).astype(np.float32)

strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 30).astype(np.float32)
maturity_range = np.linspace(30 / 365.25, 2, 25).astype(np.float32)



def compute_option_price(prices, T, K):
    discount_factor = np.exp(-r * T)
    payoffs = np.maximum(prices - K, 0)

    return np.mean(payoffs) * discount_factor


def compute_option_and_iv(idx):
    a, b, c, eta, rho, H = (
        a_samples[idx],
        b_samples[idx],
        c_samples[idx],
        eta_samples[idx],
        rho_samples[idx],
        H_samples[idx]
    )
  
    data_points = []
    xi0 = [a, b, c]
    max_maturity = np.max(maturity_range)
    prices = mc_sim(S0, n, m, r, max_maturity, xi0, eta, rho, H, whole_process=True)
    
    for T in maturity_range:
        for K in strike_range:
            prices_cropped = prices[:,min(int(n*(T/2)),n-1)]
            price = compute_option_price(prices_cropped, T, K)
            iv = implied_volatility(price, S0, K, r, T)
            if iv > 0.001 and iv < 3.0:
                data_points.append({
                    'a': xi0[0],
                    'b': xi0[1],
                    'c': xi0[2],
                    'eta': eta,
                    'rho': rho,
                    'H': H,
                    'strike': K,
                    'maturity': T,
                    'implied_volatility': iv
                })
    return data_points 

results = Parallel(n_jobs=2)(delayed(compute_option_and_iv)(i) for i in tqdm(range(num_samples), desc="Computing Volatility Grids"))

dataset = [point for sublist in results for point in sublist]

dataset_df = pd.DataFrame(dataset)
dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

dataset_df.to_csv('rbergomi_dataset_3.csv', index=False)
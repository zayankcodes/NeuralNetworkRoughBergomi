import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import least_squares


ticker = "MSFT"
stock = yf.Ticker(ticker)
underlying_price = stock.history(period="1d")["Close"].iloc[-1]

expiration_dates = stock.options
expiration_dates_dt = [datetime.strptime(date, '%Y-%m-%d') for date in expiration_dates]
desired_maturities_months = list(range(1, 25))
current_date = datetime.now()

def find_closest_expiration(target_date, expiration_dates):
    closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
    return closest_date

selected_expiration_dates = []
selected_expiration_dates_str = []

for months in desired_maturities_months:
    target_date = current_date + timedelta(days=30 * months)
    closest_date = find_closest_expiration(target_date, expiration_dates_dt)
    closest_date_str = closest_date.strftime('%Y-%m-%d')

    if closest_date_str not in selected_expiration_dates_str:
        selected_expiration_dates.append(closest_date)
        selected_expiration_dates_str.append(closest_date_str)

options_data_list = []

for exp_date in expiration_dates:
    try:
        options_chain = stock.option_chain(exp_date)
    except Exception:
        continue

    calls = options_chain.calls
    puts = options_chain.puts

    calls["expiration_date"] = exp_date
    calls["option_type"] = "call"
    puts["expiration_date"] = exp_date
    puts["option_type"] = "put"

    calls["underlying_price"] = underlying_price
    puts["underlying_price"] = underlying_price

    options_data = pd.concat([calls, puts], ignore_index=True)
    options_data_list.append(options_data)

if options_data_list:
    options_data = pd.concat(options_data_list, ignore_index=True)
else:
    exit()

options_data = options_data[
    ["underlying_price", "strike", "expiration_date", "option_type",
     "bid", "ask", "volume", "openInterest", "lastPrice", "impliedVolatility"]
]
options_data.rename(columns={"openInterest": "open_interest", "lastPrice": "last_price", 
                             "impliedVolatility": "implied_volatility"}, inplace=True)
options_data['expiration_date'] = pd.to_datetime(options_data['expiration_date'])
options_data['T'] = (options_data['expiration_date'] - current_date).dt.total_seconds() / (365.25 * 24 * 3600)
options_data = options_data[options_data['T'] > 0]
options_data['implied_volatility'] = pd.to_numeric(options_data['implied_volatility'], errors='coerce')
options_data = options_data.dropna(subset=['implied_volatility'])

iv_mean = options_data['implied_volatility'].mean()
iv_std = options_data['implied_volatility'].std()
options_data = options_data[
    (options_data['implied_volatility'] >= iv_mean - 3 * iv_std) &
    (options_data['implied_volatility'] <= iv_mean + 3 * iv_std)
]

options_data['moneyness'] = options_data['strike'] / options_data['underlying_price']
options_data = options_data[options_data['option_type'] == 'call']

options_data = options_data[
    (options_data['moneyness'] >= 0.8) & (options_data['moneyness'] <= 1.2) &
    (options_data['T'] >= 30 / 365.25) & (options_data['T'] <= 2)
]

options_data.reset_index(drop=True, inplace=True)

dataset_df = pd.DataFrame(options_data)

dataset_df.to_csv('yfinance_dataset.csv', index=False)
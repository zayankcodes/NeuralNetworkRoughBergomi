
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution, least_squares
from scipy.interpolate import SmoothBivariateSpline
from joblib import load
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


options_data = pd.read_csv('yfinance_dataset.csv')

class RoughBergomiNet(nn.Module):
    def __init__(self):
        super(RoughBergomiNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)


model = RoughBergomiNet()
model.load_state_dict(torch.load("rbergomi_model.pth", map_location=torch.device("cpu")))
model.eval() 

X_scaler = load("X_scaler.joblib")

underlying_price = options_data['underlying_price'].iloc[0]
options_data['strike'] = options_data['strike'] / underlying_price

strike_grid = np.linspace(options_data['strike'].min(), options_data['strike'].max(), 30)
maturity_grid = np.linspace(options_data['T'].min(), options_data['T'].max(), 30)  
strike_mesh, maturity_mesh = np.meshgrid(strike_grid, maturity_grid)


spline = SmoothBivariateSpline(
    options_data['strike'],
    options_data['T'],  
    options_data['implied_volatility'], 
    kx=3, ky=3
)
target_iv_surface = spline.ev(strike_mesh.ravel(), maturity_mesh.ravel()).reshape(len(maturity_grid), len(strike_grid))

import numpy as np
import torch
def calibration_loss(params, model, target_iv_surface, strike_grid, maturity_grid, X_scaler):

    a, b, c, eta, rho, H = params

    K, T = np.meshgrid(strike_grid, maturity_grid)
    K_flat = K.ravel()
    T_flat = T.ravel()

    param_sets = np.column_stack([
        np.full_like(K_flat, a),
        np.full_like(K_flat, b),
        np.full_like(K_flat, c),
        np.full_like(K_flat, eta),
        np.full_like(K_flat, rho),
        np.full_like(K_flat, H),
        K_flat,
        T_flat
    ])


    param_scaled = X_scaler.transform(param_sets)

    param_tensor = torch.tensor(param_scaled, dtype=torch.float32)
    

    pred_iv = model(param_tensor).squeeze()
    

    pred_iv_surface = pred_iv.view(*K.shape)
    

    target_iv_tensor = torch.tensor(target_iv_surface, dtype=torch.float32)
    
    loss = torch.sum((pred_iv_surface - target_iv_tensor) ** 2)
    
    return loss.item()  


bounds = [(0.002, 0.1),    # a
             (0.002, 0.1),    # b
             (0.002, 0.1),    # c
            (0.5, 4.0),   # eta
            (-0.95, -0.1),   # rho
            (0.025, 0.5)]   # H

result_DE = differential_evolution(
    calibration_loss,
    bounds=bounds,
    args=(model, target_iv_surface, strike_grid, maturity_grid, X_scaler),
    strategy='best1bin',
    maxiter=1000,
    popsize=15,
    tol=1e-6,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    polish=True,
    disp=True
)


if result_DE.success:
    print("\nOptimization succeeded!")
    params_list = [float(x) for x in result_DE.x]
    print(params_list)
    print(f"Calibration Loss: {result_DE.fun:.4f}\n")
else:
    print("\nOptimization failed:", result_DE.message)
    calibrated_params = None



def generate_nn_iv_surface(model, params, strike_grid, maturity_grid, X_scaler):
    """
    Generate the implied volatility surface from the trained NN model given calibrated parameters.
    params should be [a, b, c, eta, rho, H].
    """
    a, b, c, eta, rho, H = params
    
    # Create a grid of strikes (K) and maturities (T)
    K, T = np.meshgrid(strike_grid, maturity_grid)
    K_flat = K.ravel()
    T_flat = T.ravel()
    

    param_sets = np.column_stack([
        np.full_like(K_flat, a),
        np.full_like(K_flat, b),
        np.full_like(K_flat, c),
        np.full_like(K_flat, eta),
        np.full_like(K_flat, rho),
        np.full_like(K_flat, H),
        K_flat,
        T_flat
    ])
    
    param_scaled = X_scaler.transform(param_sets)
    

    param_tensor = torch.tensor(param_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        pred_iv = model(param_tensor).squeeze()
    
    iv_surface = pred_iv.view(*K.shape).numpy()
    return iv_surface

if result_DE.success:
    total_grid_points = len(strike_grid) * len(maturity_grid)
    

    mse = result_DE.fun / total_grid_points
    
    rmse = np.sqrt(mse)
    print(f"Calibration Loss (SSE): {result_DE.fun:.4f}")
    print(f"RMSE: {rmse:.6f}")


    nn_iv_surface = generate_nn_iv_surface(
        model=model,
        params=params_list,
        strike_grid=strike_grid,
        maturity_grid=maturity_grid,
        X_scaler=X_scaler
    )

    residuals = nn_iv_surface - target_iv_surface


else:
    print("\nOptimization failed:", result_DE.message)
    calibrated_params = None
import numpy as np
import torch
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf_market = ax1.plot_surface(
    strike_mesh,
    maturity_mesh,
    target_iv_surface,
    cmap='viridis',
    edgecolor='none'
)
ax1.set_title("Market (Spline) IV Surface")
ax1.set_xlabel("Strike")
ax1.set_ylabel("Maturity")
ax1.set_zlabel("Implied Volatility")

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
surf_nn = ax2.plot_surface(
    strike_mesh,
    maturity_mesh,
    nn_iv_surface,
    cmap='plasma',
    edgecolor='none'
)
ax2.set_title("NN Calibrated IV Surface")
ax2.set_xlabel("Strike")
ax2.set_ylabel("Maturity")
ax2.set_zlabel("Implied Volatility")

ax3 = fig.add_subplot(1, 3, 3)

res_levels = 50  
contour_plot = ax3.contourf(
    strike_mesh,
    maturity_mesh,
    residuals,
    levels=res_levels,
    cmap='coolwarm'
)
ax3.set_title("Residuals (NN - Market)")
ax3.set_xlabel("Strike")
ax3.set_ylabel("Maturity")


cbar_res = plt.colorbar(contour_plot, ax=ax3)
cbar_res.set_label("IV Difference")

plt.tight_layout()
plt.show()
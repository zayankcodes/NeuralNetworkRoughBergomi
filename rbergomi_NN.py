import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd



combined_df = pd.read_csv('rbergomi_dataset_final.csv')



S0 = 1
strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 30)
maturity_range = np.linspace(30 / 365.25, 2, 25)


features = ['a', 'b', 'c', 'eta', 'rho', 'H', 'strike', 'maturity']
target = 'implied_volatility'

X = combined_df[features].values
y = combined_df[target].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_scaler = RobustScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


dump(X_scaler, "X_scaler.joblib")


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()  

num_epochs = 100
patience = 10
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(num_epochs):

    model.train()
    train_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_losses.append(loss.item())
    
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
   
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict() 
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            model.load_state_dict(best_model_state)
            break


if best_model_state is not None:
    model.load_state_dict(best_model_state)

torch.save(model.state_dict(), "rbergomi_model.pth")


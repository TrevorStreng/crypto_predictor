import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

data = pd.read_csv('eth_prices.csv')
data['Open'] = data['Open'].str.replace(',', '').astype(float)
prices = data['Open'].values.reshape(-1, 1)

# scale values to between 0 and 1 based on the column min and max
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(prices)

def create_sequences(data, seq_length):
  xs, ys = [], []
  for i in range(len(data) - seq_length):
    x = data[i:i+seq_length]
    y = data[i+seq_length]
    xs.append(x)
    ys.append(y)
  return np.array(xs), np.array(ys)

sequence_length = 25
X, y = create_sequences(scaled_data, sequence_length)


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

class CryptoDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
  
dataset = CryptoDataset(X, y)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# I think I need more data for this to be relevant
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(LSTMModel, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out
  
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 50

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
  for batch_X, batch_y in train_loader:
    batch_X = batch_X.view(-1, sequence_length, input_size)

    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item(): .4f}')

model.eval()
test_preds = []
actuals = []

with torch.no_grad():
  for barch_X, batch_y in test_loader:
    batch_X = batch_X.view(-1, sequence_length, input_size)
    preds = model(batch_X)
    test_preds.append(preds)
    actuals.append(batch_y)

test_preds = torch.cat(test_preds).cpu().numpy().reshape(-1, 1)
actuals = torch.cat(actuals).cpu().numpy().reshape(-1, 1)

test_preds = scaler.inverse_transform(test_preds)
actuals = scaler.inverse_transform(actuals)

plt.plot(actuals, label='Actual Prices')
plt.plot(test_preds, label='Predicted Prices')
plt.legend()
plt.show()
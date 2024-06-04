import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ST_Encoder import ST_Enc
from args import parse_args
import os
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


batch_size = 64
num_epochs = 5000
learning_rate = 0.0001
patience = 30

args = parse_args()
data = np.load('/home/zhangmin/toby/UrbanGPT/data/discharge/data_encoder.npy')
N, T, F = data.shape
print(f"Data shape: N={N}, T={T}, F={F}")
mask = ~np.isnan(data)
data = np.nan_to_num(data, nan=0.0)
data_mean = np.mean(data, axis=(1, 2), keepdims=True)
data_std = np.std(data, axis=(1, 2), keepdims=True)
data_std[data_std == 0] = 1  # 防止除以零
data = (data - data_mean) / data_std

if np.isnan(data).any():
    print("Data contains NaN values after interpolation.")
else:
    print("No NaN values in data after interpolation.")

input_window = args.input_window
output_window = args.output_window
data_x = []
data_y = []
mask_x = []
mask_y = []

for t in range(T - input_window - output_window + 1):
    x = data[:, t:t+input_window, :].transpose(1, 0, 2)
    y = data[:, t+input_window:t+input_window+output_window, :].transpose(1, 0, 2)
    mask_x.append(mask[:, t:t+input_window, :].transpose(1, 0, 2))
    mask_y.append(mask[:, t+input_window:t+input_window+output_window, :].transpose(1, 0, 2))
    data_x.append(x)
    data_y.append(y)

data_x = np.array(data_x)
data_y = np.array(data_y)
mask_x = np.array(mask_x)
mask_y = np.array(mask_y)

print(f"Data_x shape: {data_x.shape}, Data_y shape: {data_y.shape}")
assert data_x.shape[1] == input_window, f"input sequence length not equal to preset sequence length, expected {input_window}, but got {data_x.shape[1]}"

data_x = torch.tensor(data_x, dtype=torch.float32)
data_y = torch.tensor(data_y, dtype=torch.float32)
mask_x = torch.tensor(mask_x, dtype=torch.float32)
mask_y = torch.tensor(mask_y, dtype=torch.float32)

train_dataset = TensorDataset(data_x, data_y, mask_x, mask_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = ST_Enc(args, dim_in=F, dim_out=args.output_dim).to('cuda')


def masked_mae_loss(output, target, mask):
    mask = mask.to(torch.bool)
    return torch.abs(output[mask] - target[mask]).mean()


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
checkpoint_path = '/home/zhangmin/toby/UrbanGPT/checkpoints/st_encoder/checkpoint.pt'
early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_x, batch_y, batch_mask_x, batch_mask_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_x, batch_y, batch_mask_x, batch_mask_y = batch_x.to('cuda'), batch_y.to('cuda'), batch_mask_x.to('cuda'), batch_mask_y.to('cuda')

        optimizer.zero_grad()
        output, _ = model(batch_x)
        loss = masked_mae_loss(output, batch_y, batch_mask_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
    early_stopping(avg_epoch_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

model.load_state_dict(torch.load(checkpoint_path))
final_checkpoint_dir = '/home/zhangmin/toby/UrbanGPT/checkpoints/st_encoder'
os.makedirs(final_checkpoint_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(final_checkpoint_dir, 'pretrain_stencoder.pth'))

print("Model saved successfully!")

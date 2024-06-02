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


# 定义超参数
batch_size = 64
num_epochs = 5000
learning_rate = 0.0001
patience = 30

# 解析参数
args = parse_args()

# 加载数据集
data = np.load('/home/zhangmin/toby/UrbanGPT/data/discharge/data_encoder.npy')
N, T, F = data.shape

# 打印加载的数据集形状
print(f"Data shape: N={N}, T={T}, F={F}")

# 数据标准化处理
data_mean = data.mean(axis=(1, 2), keepdims=True)
data_std = data.std(axis=(1, 2), keepdims=True)
data = (data - data_mean) / data_std

# 将数据集转换为Tensor并创建DataLoader
input_window = args.input_window
output_window = args.output_window

# 创建输入和输出数据
data_x = []
data_y = []
for t in range(T - input_window - output_window + 1):
    data_x.append(data[:, t:t+input_window, :].transpose(1, 0, 2))
    data_y.append(data[:, t+input_window:t+input_window+output_window, :].transpose(1, 0, 2))

# 转换为NumPy数组再转换为Tensor
data_x = np.array(data_x)
data_y = np.array(data_y)

# 打印转换后的数据形状
print(f"Data_x shape: {data_x.shape}, Data_y shape: {data_y.shape}")

# 确保输入数据的时间步长与模型的输入窗口大小匹配
assert data_x.shape[1] == input_window, f"input sequence length not equal to preset sequence length, expected {input_window}, but got {data_x.shape[1]}"

data_x = torch.tensor(data_x, dtype=torch.float32)
data_y = torch.tensor(data_y, dtype=torch.float32)

train_dataset = TensorDataset(data_x, data_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = ST_Enc(args, dim_in=F, dim_out=args.output_dim).to('cuda')

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 初始化早停机制
checkpoint_path = '/home/zhangmin/toby/UrbanGPT/checkpoints/st_encoder/checkpoint.pt'
early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

# 训练模型
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to('cuda'), batch_y.to('cuda')

            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    # 检查早停条件
    early_stopping(avg_epoch_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# 加载最佳模型权重
model.load_state_dict(torch.load(checkpoint_path))

# 保存模型权重
final_checkpoint_dir = '/home/zhangmin/toby/UrbanGPT/checkpoints/st_encoder'
os.makedirs(final_checkpoint_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(final_checkpoint_dir, 'pretrain_stencoder.pth'))

print("Model saved successfully!")

# 预测并还原真实值


def inverse_transform(data, mean, std):
    return data * std + mean


model.eval()
with torch.no_grad():
    sample_x, sample_y = next(iter(train_loader))
    sample_x = sample_x.to('cuda')
    predicted_y, _ = model(sample_x)

    # 反标准化
    predicted_y = predicted_y.cpu().numpy()
    sample_y = sample_y.numpy()

    predicted_y = inverse_transform(predicted_y, data_mean, data_std)
    sample_y = inverse_transform(sample_y, data_mean, data_std)

    print("Predicted values (first batch):")
    print(predicted_y)
    print("True values (first batch):")
    print(sample_y)

# %%
from trainer import torch_trainer
from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels  # type: ignore
from modules.transformer import SelfAttentionBlocks, MLP, CrossAttentionBlocks, sinusoidal_positional_encoding  # type: ignore
import os
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------- COP Loss -----------------------------


class COPLoss(nn.Module):
    """
    COP(y_true, y_pred) = sum((y_true - y_pred)^2)
                         / (batch_size * Var(flatten(y_true)))
    Mirrors the TensorFlow version:
      var_true = batch_size * reduce_variance(flatten(y_true))
      data_loss = sum(squared_error) / var_true
    """

    def __init__(self):
        super(COPLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # flatten everything
        yt = y_true.view(-1)
        yp = y_pred.view(-1)
        # numerator: sum of squared errors
        sqr_err = (yt - yp).pow(2).sum()
        # population variance of flattened y_true
        var_flat = yt.var(unbiased=False)
        # scale by batch size
        var_true = y_true.size(0) * var_flat
        return sqr_err / var_true


# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
num_samples = 5000
# PATH = '/projects/bblv/skoric/DEEPXDE_TEST_MULTIPHYSICS_MULTI_BRANCH/Data_Clean'
PATH = '../data/steel_solidification'
loads1 = np.load(f'{PATH}/disp_filtered.npy')[:num_samples]
loads2 = np.load(f'{PATH}/flux_filtered.npy')[:num_samples]
sol1 = np.load(f'{PATH}/filtered_stress_data.npy')[:num_samples]
sol2 = np.load(f'{PATH}/filtered_temp_data.npy')[:num_samples]
coords = np.load(f'{PATH}/xy_train_testing.npy')
sol1 = sol1[:, -1, :-1]
sol2 = sol2[:, -1, :-1]
coords = coords.astype(np.float32)
loads1 = loads1.astype(np.float32)
loads2 = loads2.astype(np.float32)
sol1 = sol1.astype(np.float32)
sol2 = sol2.astype(np.float32)

# %%
xy_mean, xy_std = np.mean(coords, axis=0)[None], np.std(coords, axis=0)[None]
xy_train_testing = (coords - xy_mean) / xy_std
# move xy positions to tensor on device
torch_xy = torch.from_numpy(xy_train_testing).to(device)

sigma_mean, sigma_std = np.mean(sol1), np.std(sol1)
sigma_norm = (sol1 - sigma_mean) / sigma_std

temp_mean, temp_std = np.mean(sol2), np.std(sol2)
temp_norm = (sol2 - temp_mean) / temp_std

# s_std = np.array([sigma_std, temp_std]).reshape(1, 1, 2)
# s_mean = np.array([sigma_mean, temp_mean]).reshape(1, 1, 2)
# s_norm = np.concatenate([sigma_norm[:, :, None], temp_norm[:, :, None]], axis=-1)  # (B, N,2)
s_norm = np.stack([sigma_norm, temp_norm], axis=-1)  # shape (N, nodes, 2)


def solu_inv(y):
    """
    y: numpy array of shape (samples, nodes, 2),
       where [:,:,0] is normalized stress and [:,:,1] is normalized temp.
    """
    y0 = y[..., 0] * sigma_std + sigma_mean
    y1 = y[..., 1] * temp_std + temp_mean
    return np.stack([y0, y1], axis=-1)


load1_all = loads1[:, :, None]
load2_all = loads2[:, :, None]

loads1_mean, loads1_std = np.mean(load1_all), np.std(load1_all)
loads1_norm = (load1_all - loads1_mean) / loads1_std
loads2_mean, loads2_std = np.mean(load2_all), np.std(load2_all)
loads2_norm = (load2_all - loads2_mean) / loads2_std

loads1_norm = torch.from_numpy(loads1_norm)
loads2_norm = torch.from_numpy(loads2_norm)

s_all = torch.from_numpy(s_norm)

#######
# load1_train, load1_test, load2_train, load2_test, s_train, s_test = train_test_split(
# loads1_norm, loads2_norm, s_all, test_size=0.2, random_state=2024)
# instead compute split point
N = len(s_all)
n_train = int(0.8 * N)
print("n_train = ", n_train)

load1_train, load1_test = loads1_norm[:n_train], loads1_norm[n_train:]
load2_train, load2_test = loads2_norm[:n_train], loads2_norm[n_train:]
s_train,      s_test = s_all[:n_train],       s_all[n_train:]
######

dataset_train = TensorDataset(load1_train, load2_train, s_train)
dataset_test = TensorDataset(load1_test, load2_test, s_test)

dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False)

# %%


class Branch(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, self_attn_layers=4, num_heads=4):
        super(Branch, self).__init__()
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=256,
                           batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128,
                           batch_first=True, bidirectional=False)
        # self.gru3 = nn.GRU(input_size=128, hidden_size=128,
        # batch_first=True, bidirectional=False)
        # self.gru4 = nn.GRU(input_size=128, hidden_size=256,
        # batch_first=True, bidirectional=False)
        self.time_distributed = nn.Linear(128, embed_dim)
        self.weighted_sum = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.pos_encoding = sinusoidal_positional_encoding(
            length=2048, d_model=embed_dim)[None, :, :].to(device)
        self.attention_blocks = SelfAttentionBlocks(
            width=embed_dim, heads=num_heads, layers=self_attn_layers)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        # out, _ = self.gru3(out)
        # out, _ = self.gru4(out)
        x = self.time_distributed(out)
        B, S, embed_dim = x.shape
        x = x * torch.sqrt(torch.tensor(embed_dim,
                           dtype=torch.float32, device=x.device))
        x = torch.cat([
            x,
            self.pos_encoding[:, :S, :].repeat(B, 1, 1)
        ], dim=-1)
        x = self.weighted_sum(x)
        x = self.attention_blocks(x)
        return x


class Truck(nn.Module):
    def __init__(
        self, branch: Branch, embed_dim=64,
        cross_attn_layers=4, num_heads=4,
        in_channels=2, out_channels=1,
        emd_version='nerf'
    ):
        super(Truck, self).__init__()
        self.branch = branch
        self.emd_version = emd_version
        d = position_encoding_channels(emd_version)
        self.Q_encoder = nn.Sequential(
            nn.Linear(d * in_channels, 2 * embed_dim),
            nn.SiLU(),
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.SiLU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )
        self.cross_att_blocks = CrossAttentionBlocks(
            width=embed_dim, heads=num_heads, layers=cross_attn_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.SiLU(),
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.SiLU(),
            nn.Linear(2 * embed_dim, out_channels)
        )

    def forward(self, inp_load1, inp_load2, x):
        B = inp_load1.shape[0]
        inp = torch.cat([inp_load1, inp_load2], dim=-1).to(device)
        br_out = self.branch(inp)
        x_enc = encode_position(self.emd_version, position=x).to(device)
        Q = self.Q_encoder(x_enc)
        Q = Q[None].repeat(B, 1, 1)
        out = self.cross_att_blocks(Q, br_out)
        out = self.output_proj(out)  # (B, nodes, 2)
        return out
        # return out.squeeze(-1) ### (B, nodes, 1) -> (B, nodes)


class TRAINER(torch_trainer.TorchTrainer):
    def __init__(self, models, device, filebase):
        super().__init__(models, device, filebase)

    def evaluate_losses(self, data):
        inp1, inp2, y_true = [d.to(self.device) for d in data]
        y_pred = self.models[0](inp1, inp2, torch_xy)
        # loss = nn.MSELoss()(y_true, y_pred)
        # loss = COPLoss()(y_pred, y_true)
        loss = nn.L1Loss()(y_true, y_pred)

        return loss, {'loss': loss.item()}

    def predict(self, data_loader):
        y_pred, y_true = [], []
        self.models[0].eval()
        with torch.no_grad():
            # for inp1, inp2, y_true in data_loader:
            for data in data_loader:
                inp1, inp2 = data[0].to(self.device), data[1].to(self.device)
                y_true_batch = data[2].to(self.device)
                pred = self.models[0](inp1, inp2, torch_xy)

                y_pred.append(pred.cpu().detach().numpy())
                y_true.append(y_true_batch.cpu().detach().numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        return y_pred, y_true


branch = Branch(embed_dim=64, self_attn_layers=3, num_heads=1).to(device)
snot = Truck(
    branch, embed_dim=64,
    cross_attn_layers=4, num_heads=1,
    in_channels=2, out_channels=2
).to(device)

trainer = TRAINER(snot, device, './saved_weights/test_slice_both')
optimizer = torch.optim.Adam(trainer.parameters(), lr=1e-3)
checkpoint = torch_trainer.ModelCheckpoint(
    monitor='val_loss', save_best_only=True)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.7, patience=50)
trainer.compile(
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    checkpoint=checkpoint,
    scheduler_metric_name='val_loss'
)
# history = trainer.fit(
#     dataloader_train, val_loader=dataloader_test, epochs=3000)
# trainer.save_logs()

# evaluation
import time
start_time = time.time()
y_pred_test, y_true_test = trainer.predict(dataloader_test)
print(f"Prediction time: {time.time() - start_time:.2e} seconds, each sample: {(time.time() - start_time) / len(y_true_test):.4e} seconds")

y_pred_test = solu_inv(y_pred_test)
y_true_test = solu_inv(y_true_test)


stress_true_test = y_true_test[..., 0]
stress_pred_test = y_pred_test[..., 0]

temp_true_test = y_true_test[..., 1]
temp_pred_test = y_pred_test[..., 1]

np.savez_compressed('s-not_slice_results.npz', a=temp_true_test,
                    b=stress_true_test, c=temp_pred_test, d=stress_pred_test)


# separate channels
err_test_stress = np.linalg.norm(y_true_test[..., 0] - y_pred_test[..., 0], axis=1) \
    / np.linalg.norm(y_true_test[..., 0], axis=1)
err_test_temp = np.linalg.norm(y_true_test[..., 1] - y_pred_test[..., 1], axis=1) \
    / np.linalg.norm(y_true_test[..., 1], axis=1)

print(
    f'Stress test L₂ error → mean: {err_test_stress.mean():.4f}, std: {err_test_stress.std():.4f}, max: {err_test_stress.max():.4f}')
print(
    f'Temp   test L₂ error → mean: {err_test_temp.mean():.4f},   std: {err_test_temp.std():.4f},   max: {err_test_temp.max():.4f}')

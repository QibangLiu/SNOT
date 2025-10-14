
# %%
from trainer import torch_trainer
from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels  # type: ignore
from modules.transformer import SelfAttentionBlocks, MLP, CrossAttentionBlocks, sinusoidal_positional_encoding  # type: ignore
import os
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_data = 4000

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


# %%
# data_loc = "/projects/bblv/skoric/DEEPXDE_TEST_VECTOR_DOG_BONE_PLASTIC/Data"
data_loc = "../data/dogbone"
data_t = np.load(
    data_loc+'/PEEQ.npz')['a'].astype(np.float32)[:N_data, -1, :]  # (4000,3060)
data_s = np.load(
    data_loc+'/Stress.npz')['a'].astype(np.float32)[:N_data, -1, :]  # (4000,3060)

# Scale, PEEQ
pmax = 0.6 * np.max(data_t)
flag = data_t > pmax
print('Capped ', np.sum(flag) / float(len(flag.flatten()))
      * 100, ' percent PEEQ data points')
data_t[flag] = pmax

# Cap stress
smax = 260.
flag = data_s > smax
print('Capped ', np.sum(flag) / float(len(flag.flatten()))
      * 100, ' percent stress data points')
data_s[flag] = smax


Heat_Amp = np.load(
    data_loc+'/Amp.npz')['a'].astype(np.float32)[:N_data]  # (4000,3060)

xy_train_testing = np.load(data_loc+'/Coords.npy').astype(np.float32)
xy_train_testing_org = xy_train_testing

xy_mean, xy_std = np.mean(xy_train_testing, axis=0)[None], np.std(
    xy_train_testing, axis=0)[None]
xy_train_testing = (xy_train_testing-xy_mean)/xy_std
# %%


sigma_mean, sigma_std = np.mean(data_s), np.std(data_s)
sigma_norm = (data_s-sigma_mean)/sigma_std

pe_mean, pe_std = np.mean(data_t), np.std(data_t)
pe_norm = (data_t-pe_mean)/pe_std

s_norm = np.stack([sigma_norm, pe_norm], axis=-1)  # shape (N, nodes, 2)


def solu_inv(y):
    """
    y: numpy array of shape (samples, nodes, 2),
       where [:,:,0] is normalized stress and [:,:,1] is normalized peeq
    """
    y0 = y[..., 0] * sigma_std + sigma_mean
    y1 = y[..., 1] * pe_std + pe_mean
    return np.stack([y0, y1], axis=-1)

# s_std = np.array([sigma_std, pe_std]).reshape(1, 1, 2)
# s_mean = np.array([sigma_mean, pe_mean]).reshape(1, 1, 2)
# s_norm = np.concatenate(
# [sigma_norm[:, :, None], pe_norm[:, :, None]], axis=-1)  # (B, N,2)


# def solu_inv(y):
# return y*s_std+s_mean


u0_all = Heat_Amp[:, :, None]
u0_all = torch.from_numpy(u0_all)
s_all = torch.from_numpy(s_norm)
xy_train_testing = torch.from_numpy(xy_train_testing)

N = len(s_all)
# n_train = int(0.8 * N)
n_train = int(0.8 * N)
print("n_train = ", n_train)
u0_train = u0_all[:n_train]
u0_test = u0_all[n_train:]
s_train = s_all[:n_train]
s_test = s_all[n_train:]

"""
u0_train, u0_test, s_train, s_test = train_test_split(
    u0_all, s_all, test_size=0.01, random_state=2024)
"""

dataset_train = TensorDataset(u0_train, s_train)
dataset_test = TensorDataset(u0_test, s_test)

dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=50, shuffle=False)
# %%


class Branch(nn.Module):
    def __init__(self, input_dim=1, embed_dim=64, self_attn_layers=4, emb_hidden_dims=[128, 128], num_heads=4):
        super(Branch, self).__init__()

        # input_size = input_dim = 1, only one input
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=256,
                           batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128,
                           batch_first=True, bidirectional=False)
        # self.gru3 = nn.GRU(input_size=128, hidden_size=128,
        # batch_first=True, bidirectional=False)
        # self.gru4 = nn.GRU(input_size=128, hidden_size=256,
        # batch_first=True, bidirectional=False)

        # self.time_distributed = nn.Linear(256, embed_dim)
        self.time_distributed = nn.Linear(128, embed_dim)

        # projection to embedding dimension
        # self.fc = nn.Linear(256, embed_dim)

        self.weighted_sum = nn.Linear(2 * embed_dim, embed_dim, bias=False)

        self.pos_encoding = sinusoidal_positional_encoding(
            length=2048, d_model=embed_dim)
        self.pos_encoding = self.pos_encoding[None, :, :].to(device)

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
    def __init__(self, branch,  embed_dim=64, cross_attn_layers=4, num_heads=4,
                 in_channels=2, out_channels=2,
                 emd_version="nerf"):
        super(Truck, self).__init__()
        self.branch = branch
        self.emd_version = emd_version
        d = position_encoding_channels(emd_version)

        self.Q_encoder = nn.Sequential(nn.Linear(d*in_channels, 2*embed_dim),
                                       nn.SiLU(),
                                       nn.Linear(2*embed_dim, 2*embed_dim),
                                       nn.SiLU(),
                                       nn.Linear(2*embed_dim, embed_dim)
                                       )

        self.cross_att_blocks = CrossAttentionBlocks(
            width=embed_dim, heads=num_heads, layers=cross_attn_layers)

        self.output_proj = nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),
                                         nn.SiLU(),
                                         nn.Linear(2*embed_dim, 2*embed_dim),
                                         nn.SiLU(),
                                         nn.Linear(2*embed_dim, out_channels)
                                         )

    def forward(self, inp_fun, x):
        """
        inp_fun: (B, S, in_channels)
        x: (nx, nd)
        """
        B = inp_fun.shape[0]
        br_out = self.branch(inp_fun)
        x = encode_position(self.emd_version, position=x)
        x = self.Q_encoder(x)
        x = x[None].repeat(B, 1, 1)
        x = self.cross_att_blocks(x, br_out)
        x = self.output_proj(x)  # (B, nodes, 2) ##2 for two solution fields
        return x


# %%
branch = Branch(input_dim=1, embed_dim=64, self_attn_layers=3,
                emb_hidden_dims=[128, 128], num_heads=1).to(device)
snot = Truck(branch, embed_dim=64, cross_attn_layers=4,
             num_heads=1, in_channels=2, out_channels=2).to(device)

# %%
# xinp = torch.randn(2, 101, 1)
# brout = branch(xinp)
# %%
xy_train_testing = xy_train_testing.to(device)


class TRAINER(torch_trainer.TorchTrainer):
    def __init__(self, models, device, filebase):
        super().__init__(models, device, filebase)

    def evaluate_losses(self, data):
        inp_fun = data[0].to(self.device)
        y_true = data[1].to(self.device)
        y_pred = self.models[0](inp_fun, xy_train_testing)
        # loss = nn.MSELoss()(y_true, y_pred)
        loss = nn.L1Loss()(y_true, y_pred)
        # loss = COPLoss()(y_pred, y_true)
        loss_dic = {"loss": loss.item()}
        return loss, loss_dic

    def predict(self, data_loader):
        y_pred = []
        y_true = []
        self.models[0].eval()
        with torch.no_grad():
            for data in data_loader:
                inp_fun = data[0].to(self.device)
                y_true_batch = data[1].to(self.device)
                pred = self.models[0](inp_fun, xy_train_testing)

                y_pred.append(pred.cpu().detach().numpy())
                y_true.append(y_true_batch.cpu().detach().numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        return y_pred, y_true


# %%
trainer = TRAINER(
    snot, device, "./saved_weights/test_dog_bone_gru_l1")
optimizer = torch.optim.Adam(trainer.parameters(), lr=1e-3)
checkpoint = torch_trainer.ModelCheckpoint(
    monitor="val_loss", save_best_only=True)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.7, patience=20)
trainer.compile(
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    checkpoint=checkpoint,
    scheduler_metric_name="val_loss",
)
# h = trainer.fit(dataloader_train, val_loader=dataloader_test,
#                 epochs=3000)
# trainer.save_logs()

# %%
trainer.load_weights(device=device)
start_time = time.time()
y_pred, y_true = trainer.predict(dataloader_test)
print(f"Prediction time: {time.time() - start_time:.2f} seconds, each sample: {(time.time() - start_time) / len(y_pred):.4f} seconds")
y_pred = solu_inv(y_pred)
y_true = solu_inv(y_true)

error_s = np.linalg.norm(y_true[:, :, 0] - y_pred[:, :, 0], axis=1) / (np.linalg.norm(
    y_true[:, :, 0], axis=1) + 1e-8)

print(
    f"Mean L2 error (stress) for test data: {np.mean(error_s)}, std: {np.std(error_s)}, max: {np.max(error_s)}")


abs_error_p = np.abs(y_pred[:, :, 1] - y_true[:, :, 1])
# mean absolute error of each row, sample
abs_error_p_samp = np.mean(abs_error_p, axis=1)
# mean absolute error for all rows (samples)
print('mean of absolute error of Peeq: {:.4e}'.format(
    np.mean(abs_error_p_samp)))
print('std of absolute error of Peeq: {:.4e}'.format(np.std(abs_error_p_samp)))
print('max of absolute error of Peeq: {:.4e}'.format(np.max(abs_error_p_samp)))

stress_true_test = y_true[..., 0]
stress_pred_test = y_pred[..., 0]

peeq_true_test = y_true[..., 1]
peeq_pred_test = y_pred[..., 1]

np.savez_compressed('s-not_dog_bone_results.npz', a=peeq_true_test, b=stress_true_test,
                    c=peeq_pred_test, d=stress_pred_test, e=xy_train_testing_org)

# %%

# src/models.py
import torch
import torch.nn as nn


class MultiChannelFiLM(nn.Module):
    def __init__(self, static_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim * 6)
        )
        self.latent_dim = latent_dim

    def forward(self, x_static):
        out = self.net(x_static)
        soh_g, soh_b = out[:, :self.latent_dim], out[:, self.latent_dim: self.latent_dim * 2]
        pos_g, pos_b = out[:, self.latent_dim * 2: self.latent_dim * 3], out[
            :, self.latent_dim * 3: self.latent_dim * 4]
        neg_g, neg_b = out[:, self.latent_dim * 4: self.latent_dim * 5], out[:, self.latent_dim * 5:]
        return soh_g, soh_b, pos_g, pos_b, neg_g, neg_b


class AdaptiveChannelEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, feat_dropout_prob):
        super().__init__()
        self.feat_dropout = nn.Dropout2d(p=feat_dropout_prob)
        self.gru = nn.GRU(input_dim, latent_dim, batch_first=True)

    def forward(self, x_seq):
        x_seq = x_seq.transpose(1, 2).unsqueeze(-1)
        x_seq = self.feat_dropout(x_seq).squeeze(-1).transpose(1, 2)
        _, h = self.gru(x_seq)
        return h.squeeze(0)


class TrajectoryReconstructionDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim * 3 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z_soh, z_pos, z_neg, k):
        inputs = torch.cat([z_soh, z_pos, z_neg, k], dim=1)
        return self.decoder_fc(inputs)


class TAPMNetModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.soh_encoder = AdaptiveChannelEncoder(1, cfg.latent_dim, cfg.feature_dropout)
        self.pos_encoder = AdaptiveChannelEncoder(cfg.pos_features_num, cfg.latent_dim, cfg.feature_dropout)
        self.neg_encoder = AdaptiveChannelEncoder(cfg.neg_features_num, cfg.latent_dim, cfg.feature_dropout)
        self.film = MultiChannelFiLM(cfg.static_features, cfg.latent_dim)
        self.decoder = TrajectoryReconstructionDecoder(cfg.latent_dim)

    def forward(self, x_history_soh, x_history_pos, x_history_neg, x_static, k_future):
        z_soh = self.soh_encoder(x_history_soh)
        z_pos = self.pos_encoder(x_history_pos)
        z_neg = self.neg_encoder(x_history_neg)
        soh_g, soh_b, pos_g, pos_b, neg_g, neg_b = self.film(x_static)

        z_soh_modulated = soh_g * z_soh + soh_b
        z_pos_modulated = pos_g * z_pos + pos_b
        z_neg_modulated = neg_g * z_neg + neg_b

        soh_pred = self.decoder(z_soh_modulated, z_pos_modulated, z_neg_modulated, k_future)
        return soh_pred
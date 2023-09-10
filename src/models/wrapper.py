import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, backbone, feature_dim=25, hidden_dim=25, use_features=False):
        super().__init__()

        self.backbone = backbone
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

        self.use_features = use_features
        if use_features:
            self.fc = nn.Linear(self.backbone.rep_dim + hidden_dim, 2)
        else:
            self.fc = nn.Linear(self.backbone.rep_dim, 2)

    def forward(self, x, features=None):
        reps = self.backbone(x)

        if self.use_features:
            feature_reps = self.feature_fc(features)
            reps = torch.cat([reps, feature_reps], 1)

        pred = self.fc(reps)
        return pred

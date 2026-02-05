import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class DrugEmbedding(nn.Module):
    def __init__(self, embedding_path, out_dim):
        super().__init__()
        if embedding_path.endswith('.pt'):
            emb = torch.load(embedding_path)
        else:
            import numpy as np
            emb = torch.from_numpy(np.load(embedding_path))
        assert emb.size(1) == out_dim
        self.register_buffer('emb', emb)

    def forward(self, idx):
        return self.emb[idx]


class CellEmbedding(nn.Module):
    def __init__(self, exp_dim, path_dim, out_dim):
        super().__init__()
        self.exp_fc = nn.Sequential(
            nn.Linear(exp_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU()
        )
        self.path_fc = nn.Sequential(
            nn.Linear(path_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU()
        )

    def forward(self, exp, path):
        x1 = self.exp_fc(exp)
        x2 = self.path_fc(path)
        return torch.stack([x1, x2], dim=1)


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, dropout=0.5, k=3):
        super().__init__()
        self.k = k
        self.v_net = nn.Linear(v_dim, h_dim * k)
        self.q_net = nn.Linear(q_dim, h_dim * k)
        self.h_mat = nn.Parameter(torch.randn(1, h_out, 1, h_dim * k))
        self.h_bias = nn.Parameter(torch.zeros(1, h_out, 1, 1))
        self.bn = nn.BatchNorm1d(h_dim)

    def forward(self, v, q):
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        att = torch.einsum('xhyk,bvk,bqk->bhvq', self.h_mat, v_, q_) + self.h_bias
        logits = torch.einsum('bvk,bvq,bqk->bk', v_, att[:, 0], q_)
        logits = self.bn(logits)
        return logits, att


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        layers = []
        for i in range(len(hidden_dim) - 1):
            layers += [
                nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                nn.BatchNorm1d(hidden_dim[i + 1]),
                nn.ReLU()
            ]
        self.fc1 = nn.Linear(in_dim, hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])
        self.hidden = nn.Sequential(*layers)
        self.fc2 = nn.Linear(hidden_dim[-1], 1)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.hidden(x)
        return self.fc2(x)


class ICIscbluk(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.drug_embedding = DrugEmbedding(args.drug_emb_path, args.drug_out_dim)
        self.cell_embedding = CellEmbedding(
            args.cell_exp_dim, args.cell_path_dim, args.cell_out_dim
        )
        self.ban = BANLayer(
            args.drug_out_dim, args.cell_out_dim,
            args.mlp_in_dim, args.ban_heads,
            args.ban_dropout, args.ban_k
        )
        self.mlp = MLP(args.mlp_in_dim, args.mlp_hidden_dim)

    def forward(self, drug, cell):
        v = self.drug_embedding(drug)
        q = self.cell_embedding(cell[0], cell[1])
        f, att = self.ban(v, q)
        logit = self.mlp(f).squeeze(1)
        return logit, att

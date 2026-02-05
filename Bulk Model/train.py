import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

from args import get_args
from model import ICIscbluk
from dataset import DrugCellDataset
from utils import set_seed, auc_score


def main():
    args = get_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrugCellDataset(train_drug, train_exp, train_path, train_y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_auc = []

    for fold, (tr, va) in enumerate(kf.split(dataset)):
        model = ICIscbluk(args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        tr_loader = DataLoader(Subset(dataset, tr),
                               batch_size=args.batch_size, shuffle=True)
        va_loader = DataLoader(Subset(dataset, va),
                               batch_size=args.batch_size)

        best_auc = 0

        for epoch in range(args.epochs):
            model.train()
            for drug, cell, y in tr_loader:
                drug, y = drug.to(device), y.to(device)
                cell = (cell[0].to(device), cell[1].to(device))
                optimizer.zero_grad()
                logit, _ = model(drug, cell)
                loss = criterion(logit, y)
                loss.backward()
                optimizer.step()

            model.eval()
            logits, labels = [], []
            with torch.no_grad():
                for drug, cell, y in va_loader:
                    drug = drug.to(device)
                    cell = (cell[0].to(device), cell[1].to(device))
                    logit, _ = model(drug, cell)
                    logits.append(logit)
                    labels.append(y)

            auc = auc_score(torch.cat(logits), torch.cat(labels))
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(),
                           f"checkpoints/fold{fold}_best.pt")

        fold_auc.append(best_auc)

    best_fold = int(np.argmax(fold_auc))
    with open("checkpoints/best_fold.txt", "w") as f:
        f.write(str(best_fold))

    print("Best fold:", best_fold)
    print("Fold AUCs:", fold_auc)


if __name__ == "__main__":
    main()

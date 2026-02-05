import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from args import get_args
from model import ICIscbluk
from dataset import DrugCellDataset


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("checkpoints/best_fold.txt") as f:
        best_fold = f.read().strip()

    model = ICIscbluk(args).to(device)
    model.load_state_dict(
        torch.load(f"checkpoints/fold{best_fold}_best.pt")
    )
    model.eval()

    test_set = DrugCellDataset(test_drug, test_exp, test_path, test_y)
    loader = DataLoader(test_set, batch_size=args.batch_size)

    probs, labels = [], []

    with torch.no_grad():
        for drug, cell, y in loader:
            drug = drug.to(device)
            cell = (cell[0].to(device), cell[1].to(device))
            logit, _ = model(drug, cell)
            probs.append(torch.sigmoid(logit).cpu())
            labels.append(y)

    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, probs > 0.5)

    print(f"External AUC: {auc:.4f}")
    print(f"External ACC: {acc:.4f}")

    np.save("results/probs.npy", probs)
    np.save("results/labels.npy", labels)


if __name__ == "__main__":
    main()

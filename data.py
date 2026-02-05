from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset

class PatientFeatureDataset(Dataset):


    def __init__(
        self,
        csv_path: str,
        patient_col: str = "Patient_id",
        label_col: str = "label",
        feature_start_col: int = 1,
        feature_end_col: int = -1,
    ):

        super().__init__()

        df = pd.read_csv(csv_path)

        self.patient_ids = df[patient_col].values
        self.x = torch.tensor(
            df.iloc[:, feature_start_col:feature_end_col].values,
            dtype=torch.float
        )
        self.y = torch.tensor(
            df[label_col].values,
            dtype=torch.float
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.x[idx], self.y[idx]

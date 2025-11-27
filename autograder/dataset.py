from __future__ import annotations

import random

from bidict import bidict
import pandas as pd
from torch.utils.data import Dataset, Subset

ENRON_LABEL_INDEX_MAP = bidict({
    "ham": 0,
    "spam": 1,
    "unknown": -1
})

class CPEN455_2025_W1_Dataset(Dataset):
    def __init__(
        self,
        csv_path = None
    ) -> None:
        
        self._csv_path = csv_path

        frame = pd.read_csv(csv_path)

        expected_columns = {"Index", "Subject", "Message", "Spam/Ham"}
        missing_columns = expected_columns.difference(frame.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV file missing required columns: {missing_str}")
        
        frame["Subject"] = frame["Subject"].fillna("").astype(str)
        frame["Message"] = frame["Message"].fillna("").astype(str)
        frame["Spam/Ham"] = (
            frame["Spam/Ham"].fillna("").astype(str).str.strip().str.lower()
        )

        self._index = frame["Index"].tolist()
        self._subjects = frame["Subject"].tolist()
        self._messages = frame["Message"].tolist()
        self._labels = frame["Spam/Ham"].tolist()

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(
        self, index
    ):
        data_index = self._index[index]
        subject = self._subjects[index]
        message = self._messages[index]
        label_index = int(self._labels[index])

        return data_index, subject, message, label_index
    
def prepare_subset(dataset, num_training_samples, ratio_spam=0.5, return_remaining=False):
    # Separate spam and ham samples
    spam_indices = [i for i, (_, _, _, label) in enumerate(dataset) if label == 1]
    ham_indices = [i for i, (_, _, _, label) in enumerate(dataset) if label == 0]

    # Ensure equal number of spam and ham samples
    num_spam = int(num_training_samples * ratio_spam)
    num_ham = num_training_samples - num_spam
    
    selected_spam_indices = random.sample(spam_indices, num_spam)
    selected_ham_indices = random.sample(ham_indices, num_ham)

    # Combine and shuffle the selected indices
    selected_indices = selected_spam_indices + selected_ham_indices
    random.shuffle(selected_indices)

    if return_remaining:
        remaining_indices = list(set(range(len(dataset))) - set(selected_indices))
        return Subset(dataset, selected_indices), Subset(dataset, remaining_indices)
    else:
        return Subset(dataset, selected_indices)



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = CPEN455_2025_W1_Dataset(csv_path='cpen455_released_datasets/train_val_subset.csv')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    num_ham = 0
    for idx in range(len(dataset)):
        index, subject, message, label_index = dataset[idx]
        
        if label_index == 0:
            num_ham += 1
            
        print("=" * 200)
        print(f"Email {idx + 1}/{len(dataset)}: \n")
        print("-" * 200)
        print(f"subject='{subject}'\n")
        print("-" * 200)
        print(f"label_index={label_index} \n")
        print("=" * 200)
        
    print(f"Number of ham emails: {num_ham} out of {len(dataset)}")
    
    # for batch in dataloader:
    #     index, subjects, messages, labels = batch
    #     print(f"Index: {index}")
    #     print(f"Subjects: {subjects}")
    #     print(f"Messages: {messages}")
    #     print(f"Labels: {labels}")
    #     pdb.set_trace()

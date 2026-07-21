import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SEED = 42

train_df = pd.read_csv(r"D:\Bharat\College\Projects\distillation-project\knowledge-distillation-comparison\data\fashion-mnist_distillation_train.csv")

test_df = pd.read_csv(r"D:\Bharat\College\Projects\distillation-project\knowledge-distillation-comparison\data\fashion-mnist_test.csv")

X_train = train_df.iloc[:,1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:,1:].values
y_test = test_df.iloc[:, 0].values


# scale 0-255 pixel range, for stable training
X_train = X_train/255.0
X_test = X_test/255.0


class CustomDataset(Dataset):

    def __init__(self, features, labels):

        self.features = torch.tensor(features, dtype = torch.float32).reshape(-1,1,28,28)
        self.labels = torch.tensor(labels, dtype = torch.long)

    def __len__(self):

        return len(self.features)

    def __getitem__(self, index):

        return self.features[index], self.labels[index]
    


def get_dataloaders(batch_size=32):
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

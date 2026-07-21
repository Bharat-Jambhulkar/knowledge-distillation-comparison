import torch.nn as nn

class StudentANN(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),  # ensure input is flattened

            # Hidden Layer 1
            nn.Linear(input_dim, 144),
            nn.BatchNorm1d(144),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Hidden Layer 2
            nn.Linear(144, 144),
            nn.BatchNorm1d(144),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Hidden Layer 3
            nn.Linear(144, 144),
            nn.BatchNorm1d(144),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Output Layer
            nn.Linear(144, output_dim)
        )

    def forward(self, x):
        return self.model(x)
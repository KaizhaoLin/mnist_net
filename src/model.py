import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=588, out_features=160),
            nn.ReLU(),
            nn.Linear(in_features=160, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=10)
        )

        self.net = nn.Sequential(
            self.feature_extractor,
            nn.Flatten(),
            self.classifier
        )

    def forward(self, X):
        return self.net(X)
    

if __name__ == '__main__':
    model = LeNet()
    print(model)
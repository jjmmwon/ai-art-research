import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define the feedforward neural network model
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def _train_model(model, train_loader, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).view(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()


def _evaluate_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test).view(-1)
        test_loss = criterion(test_outputs, y_test).item()
    return test_loss


def evaluate_model(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        model = FeedforwardNN(input_dim=X_train.shape[1])
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        _train_model(model, train_loader, criterion, optimizer)

        test_loss = _evaluate_model(
            model, torch.Tensor(X_test), torch.Tensor(y_test), criterion
        )
        test_scores.append(test_loss)

    return np.mean(test_scores), test_scores

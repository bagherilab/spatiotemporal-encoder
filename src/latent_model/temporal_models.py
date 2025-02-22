import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC, abstractmethod

class TemporalModel(ABC, nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(TemporalModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.criterion = nn.MSELoss()

    @abstractmethod
    def forward(self, x):
        """This method should be implemented by subclasses."""
        ...

    def fit(self, train_loader, val_loader=None, patience=5, min_delta=0.001, max_epochs=100):
        """Train model with early stopping based on validation loss."""
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        best_val_loss = float("inf")
        patience_counter = 0

        losses = []
        val_losses = []

        for _ in range(max_epochs):
            # Training
            self.train()
            total_loss = 0.0
            num_batches = 0
            for batch, _ in train_loader:
                x_batch = batch[:, :-1, :]
                y_batch = batch[:, -1, :]
                optimizer.zero_grad()
                y_pred = self(x_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()

                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            losses.append(avg_loss)

            # Eval on validation data
            self.eval()
            val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for batch, _ in val_loader:
                    x_batch = batch[:, :-1, :]
                    y_batch = batch[:, -1, :]
                    y_pred = self(x_batch)
                    val_loss += self.criterion(y_pred, y_batch).item()
                    num_val_batches += 1
            avg_val_loss = val_loss / num_val_batches
            val_losses.append(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return losses, val_losses

class RNNModel(TemporalModel):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__(input_size, hidden_size, output_size, num_layers)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LSTMModel(TemporalModel):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
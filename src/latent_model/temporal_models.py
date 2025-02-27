import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC, abstractmethod

from latent_model.sequence_loader import SequenceLoader

class TemporalModel(ABC, nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_size, 
            output_size, 
            num_layers=1, 
            dropout=0.0
        ):
        super(TemporalModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

    @abstractmethod
    def forward(self, x):
        """This method should be implemented by subclasses."""
        ...

    def fit(self, train_loader, val_loader=None, patience=5, min_delta=0.001, max_epochs=100):
        """Train model with early stopping based on validation loss."""
        
        best_val_loss = float("inf")
        patience_counter = 0

        losses = []
        val_losses = []

        for _ in range(max_epochs):
            train_loss = self.train_one_epoch(train_loader)
            losses.append(train_loss)

            val_loss = self.eval_one_epoch(val_loader)
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return losses, val_losses
    
    def train_one_epoch(self, train_loader: SequenceLoader) -> float:
        self.train()

        epoch_loss = 0.0
        optimizer = self.optimizer
        criterion = self.criterion

        for batch, _ in train_loader:
            x_batch = batch[:, :-1, :]
            y_batch = batch[:, -1, :]
            optimizer.zero_grad()

            y_pred = self(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def eval_one_epoch(self, val_loader: SequenceLoader) -> float:
        self.eval()

        val_loss = 0.0
        criterion = self.criterion

        with torch.no_grad():
            for batch, _ in val_loader:
                x_batch = batch[:, :-1, :]
                y_batch = batch[:, -1, :]
                y_pred = self(x_batch)
                val_loss += criterion(y_pred, y_batch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss

class RNNModel(TemporalModel):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__(input_size, hidden_size, output_size, num_layers)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

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

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
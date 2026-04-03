"""PyTorch RNN/LSTM models for full latent trajectory prediction and sequence classification."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from latent_model.loaders.sequence_loader import SequenceLoader


class TemporalModel(ABC, nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
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

    def _sequence_classification_targets(
        self, sample_ids: tuple | list, label_fn: Callable[[str], int], device: torch.device
    ) -> torch.Tensor:
        return torch.tensor(
            [label_fn(str(s)) for s in sample_ids],
            device=device,
            dtype=torch.long,
        )

    def train_one_epoch_sequence_classification(
        self,
        data_loader,
        label_fn: Callable[[str], int],
        device: torch.device,
    ) -> float:
        """Full-sequence input; one class label per trajectory (``classification=True`` models only)."""
        if not getattr(self, "classification", False):
            raise TypeError(
                "Use RNNModel/LSTMModel(..., classification=True) for sequence classification"
            )
        self.train()
        criterion = self.criterion
        optimizer = self.optimizer
        total = 0.0
        n_batches = 0
        for batch, sample_ids in data_loader:
            x = batch.to(device)
            y_true = self._sequence_classification_targets(sample_ids, label_fn, device)
            optimizer.zero_grad()
            logits = self.forward(x)
            if self.output_size == 1:
                loss = criterion(logits, y_true.float().unsqueeze(-1))
            else:
                loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        return total / max(n_batches, 1)

    def eval_one_epoch_sequence_classification(
        self,
        data_loader,
        label_fn: Callable[[str], int],
        device: torch.device,
    ) -> float:
        if not getattr(self, "classification", False):
            raise TypeError(
                "Use RNNModel/LSTMModel(..., classification=True) for sequence classification"
            )
        self.eval()
        criterion = self.criterion
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch, sample_ids in data_loader:
                x = batch.to(device)
                y_true = self._sequence_classification_targets(sample_ids, label_fn, device)
                logits = self.forward(x)
                if self.output_size == 1:
                    loss = criterion(logits, y_true.float().unsqueeze(-1))
                else:
                    loss = criterion(logits, y_true)
                total += loss.item()
                n_batches += 1
        return total / max(n_batches, 1)

    def fit_sequence_classification(
        self,
        train_loader,
        val_loader,
        label_fn: Callable[[str], int],
        *,
        max_epochs: int = 50,
        patience: int = 8,
        min_delta: float = 0.001,
        device: torch.device | None = None,
    ) -> tuple[list[float], list[float]]:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        best_val = float("inf")
        patience_left = patience
        train_losses: list[float] = []
        val_losses: list[float] = []
        for _ in range(max_epochs):
            tr = self.train_one_epoch_sequence_classification(train_loader, label_fn, device)
            va = self.eval_one_epoch_sequence_classification(val_loader, label_fn, device)
            train_losses.append(tr)
            val_losses.append(va)
            if va < best_val - min_delta:
                best_val = va
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        return train_losses, val_losses

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
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, classification=False):
        super().__init__(input_size, hidden_size, output_size, num_layers)
        self.classification = classification
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        if classification:
            self.criterion = nn.CrossEntropyLoss() if output_size > 1 else nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(TemporalModel):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0, classification=False
    ):
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout)
        self.classification = classification
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

        if classification:
            self.criterion = nn.CrossEntropyLoss() if output_size > 1 else nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

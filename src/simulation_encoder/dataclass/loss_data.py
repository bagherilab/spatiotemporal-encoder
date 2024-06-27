from dataclasses import dataclass, field


@dataclass
class LossData:
    """Dataclass to store loss values for training, validation, and testing from different loss functions."""

    losses_train: dict[str, list[float]] = field(default_factory=dict)
    losses_val: dict[str, list[float]] = field(default_factory=dict)
    losses_test: dict[str, float] = field(default_factory=dict)

    def add_train_loss(self, losses: dict[str, list[float]]) -> None:
        """Adds training losses to the dataclass."""
        for loss_type, values in losses.items():
            if loss_type not in self.losses_train:
                self.losses_train[loss_type] = []
            self.losses_train[loss_type].extend(values)

    def add_val_loss(self, losses: dict[str, list[float]]) -> None:
        """Adds validation losses to the dataclass."""
        for loss_type, values in losses.items():
            if loss_type not in self.losses_val:
                self.losses_val[loss_type] = []
            self.losses_val[loss_type].extend(values)

    def add_test_loss(self, losses: dict[str, float]) -> None:
        """Adds test losses to the dataclass."""
        for loss_type, value in losses.items():
            self.losses_test[loss_type] = value

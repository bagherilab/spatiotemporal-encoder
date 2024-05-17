from dataclasses import dataclass, field


@dataclass
class LossData:
    reconstruction_loss_train: list[float] = field(default_factory=list)
    reconstruction_loss_val: list[float] = field(default_factory=list)
    reconstruction_loss_test: float = 0.0
    timepoint_loss_train: list[float] = field(default_factory=list)
    timepoint_loss_val: list[float] = field(default_factory=list)
    timepoint_loss_test: float = 0.0
    combined_loss_train: list[float] = field(default_factory=list)
    combined_loss_val: list[float] = field(default_factory=list)
    combined_loss_test: float = 0.0

    def add_train_loss(self, losses: dict[str, list[float]]) -> None:
        """Adds training losses to the dataclass"""
        self.reconstruction_loss_train = losses["image"]
        self.timepoint_loss_train = losses["timepoint"]
        self.combined_loss_train = losses["combined"]

    def add_val_loss(self, losses: dict[str, list[float]]) -> None:
        """Adds validation losses to the dataclass"""
        self.reconstruction_loss_val = losses["image"]
        self.timepoint_loss_val = losses["timepoint"]
        self.combined_loss_val = losses["combined"]

    def add_test_loss(self, losses: dict[str, float]) -> None:
        """Adds test losses to the dataclass"""
        self.reconstruction_loss_test = losses["image"]
        self.timepoint_loss_test = losses["timepoint"]
        self.combined_loss_test = losses["combined"]

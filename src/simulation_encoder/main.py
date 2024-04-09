import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset

# For local imports in the module
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

from simulation_encoder.models.cnn import ConvolutionalAutoencoder
from simulation_encoder.loader import PNGLoader
from simulation_encoder.logger import ExperimentLogger

BATCH_SIZE = 10
LOAD_MODEL = False
VERBOSE = True
EXP_NAME = "autoencoder2d"
DATA_DIR = "data/ARCADE"


def run_experiment() -> None:
    """
    Run a convolutional autoencoder experiment on the ARCADE dataset.
    """
    logger = ExperimentLogger(EXP_NAME)
    dataset = PNGLoader(DATA_DIR, test_split=0.2, logger=logger, healthy_flag=False)
    train_dataset = Subset(dataset, dataset.get_train_indices())
    test_dataset = Subset(dataset, dataset.get_test_indices())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.log(
        f"Training on {len(train_dataset)} examples. Testing on {len(test_dataset)} examples."
    )
    if VERBOSE:
        print(f"Training on {len(train_dataset)} examples. Testing on {len(test_dataset)} examples.")

    autoencoder = ConvolutionalAutoencoder(
        input_shape=dataset[0].shape, out_channels=16, dim_z=50, logger=logger, verbose=VERBOSE
    )
    if LOAD_MODEL:
        autoencoder.load_state_dict(torch.load(f"saved_models/{EXP_NAME}.pth"))

    optimizer = torch.optim.Adam(autoencoder.parameters())
    loss_fn = torch.nn.MSELoss()

    losses, val_losses = autoencoder.fit(
        train_loader, epochs=10, optimizer=optimizer, loss_fn=loss_fn, val_loader=test_loader
    )

    plot_loss(losses, val_losses)

    # Save model
    logger.log(f"Trained model saved at saved_models/{EXP_NAME}.pth")
    torch.save(autoencoder.state_dict(), f"saved_models/{EXP_NAME}.pth")

    # Display sample images through trainer
    inputs = next(iter(test_loader))
    PNGLoader.display_tensor(inputs[0][0], name="figures/cancer_input.png")
    PNGLoader.display_tensor(inputs[0][1], name="figures/vasc_input.png")

    out = autoencoder.forward(inputs[0])
    PNGLoader.display_tensor(out[0][0], name="figures/cancer_output.png")
    PNGLoader.display_tensor(out[0][1], name="figures/vasc_output.png")


def plot_loss(loss: list[float], vloss: list[float]) -> None:
    """
    Plot the loss and validation loss against epochs.

    Parameters
    ----------
    loss : list[float]
        Training loss values.
    vloss : list[float]
        Validation loss values.

    """
    plt.plot(np.arange(len(loss)), loss)
    plt.plot(np.arange(len(vloss)), vloss)
    plt.legend(["Train loss", "Validation loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("figures/loss.png")


if __name__ == "__main__":
    run_experiment()

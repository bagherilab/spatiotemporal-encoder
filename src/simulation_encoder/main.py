import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from src.simulation_encoder.models.cnn import ConvolutionalAutoencoder
from src.simulation_encoder.load.loader import UnlabeledImageDataset
from src.simulation_encoder.logger import Logger

BATCH_SIZE = 10
LOAD = False
EXP_NAME = "autoencoder2d"
DATA_DIR = "data/ARCADE"


def run_experiment() -> None:
    """
    Run a convolutional autoencoder experiment on the ARCADE dataset.
    """
    logger = Logger(EXP_NAME)
    print("Logger")
    dataset = UnlabeledImageDataset(DATA_DIR, logger=logger, healthy_flag=False)
    print("Dataset")
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Training on {len(train_data)} examples. Testing on {len(test_data)} examples.")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    autoencoder = ConvolutionalAutoencoder(
        input_shape=dataset[0].shape,
        out_channels=16,
        dim_z=50,
        logger=logger,
    )
    if LOAD:
        autoencoder.load_state_dict(torch.load(f"saved_models/{EXP_NAME}.pth"))

    optimizer = torch.optim.Adam(autoencoder.parameters())
    loss_fn = torch.nn.MSELoss()

    losses, val_losses = autoencoder.fit(
        train_loader, epochs=5, optimizer=optimizer, loss_fn=loss_fn, val_loader=test_loader
    )

    plot_loss(losses, val_losses)

    # Save model
    logger.log(f"Trained model saved at saved_models/{EXP_NAME}.pth")
    torch.save(autoencoder.state_dict(), f"saved_models/{EXP_NAME}.pth")

    # Display sample images through trainer
    inputs = next(iter(test_loader))
    UnlabeledImageDataset.display_tensor(inputs[0][0], name="figures/cancer_input.png")
    # UnlabeledImageDataset.display_tensor(inputs[1], name="figures/healthy_input.png")
    UnlabeledImageDataset.display_tensor(inputs[0][1], name="figures/vasc_input.png")

    out = autoencoder.forward(inputs[0])
    UnlabeledImageDataset.display_tensor(out[0][0], name="figures/cancer_output.png")
    # UnlabeledImageDataset.display_tensor(out[0][1], name="figures/healthy_output.png")
    UnlabeledImageDataset.display_tensor(out[0][1], name="figures/vasc_output.png")


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

import numpy as np
from sklearn.model_selection import train_test_split

from cnn import ConvolutionalAutoencoder
from load.loader import UnlabeledImageDataset

import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 4

def main():
    image_dir = '../../data/test'
    dataset = UnlabeledImageDataset(image_dir)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    autoencoder = ConvolutionalAutoencoder(input_shape=dataset[0].shape)

    optimizer = torch.optim.Adam(autoencoder.parameters())
    criterion = torch.nn.MSELoss()

    autoencoder.fit(data_loader, epochs=10, )
    





if __name__ == "__main__":
    main()
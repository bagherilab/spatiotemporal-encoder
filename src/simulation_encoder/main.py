import numpy as np
from sklearn.model_selection import train_test_split

from cnn import ConvolutionalAutoencoder
from load.loader import UnlabeledImageDataset

from torch.utils.data import DataLoader

BATCH_SIZE = 4

def main():
    image_dir = '../../data/test'
    dataset = UnlabeledImageDataset(image_dir)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    autoencoder = ConvolutionalAutoencoder(input_shape=dataset[0].shape)

    for batch_images in train_loader:
        print(batch_images.shape)
        break
    
    # model = ConvolutionalAutoencoder(input_shape=(1, 256, 256))
    # model.train(train_loader, epochs=10)
    
    # model.eval()
    # for inputs, _ in test_loader:
    #     outputs = model(inputs)
    #     break
    # for i in range(4):
    #     model.display_image(inputs[i].detach().numpy())
    #     model.display_image(outputs[i].detach().numpy())





if __name__ == "__main__":
    main()
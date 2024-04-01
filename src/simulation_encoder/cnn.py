import torch
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()

    def forward(self, x):
        raise NotImplementedError("forward method should be implemented in the subclass")

    def fit(self, train_loader, epochs, optimizer, criterion):
        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, _ in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

class ConvolutionalAutoencoder(BaseCNN):
    def __init__(self, input_shape, dim_z=10):
        super(ConvolutionalAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.dim_z = dim_z
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.enc_pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_fc = nn.Linear(128 * (input_shape[1] // 8) * (input_shape[2] // 8), self.dim_z)
        
        # Decoder
        self.dec_fc = nn.Linear(self.dim_z, 128 * (input_shape[1] // 8) * (input_shape[2] // 8))
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.dec_upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.dec_upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv3 = nn.ConvTranspose2d(32, input_shape[0], kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x):
        x = torch.relu(self.enc_conv1(x))
        x = self.enc_pool1(x)
        x = torch.relu(self.enc_conv2(x))
        x = self.enc_pool2(x)
        x = torch.relu(self.enc_conv3(x))
        x = x.view(-1, 128 * (self.input_shape[1] // 8) * (self.input_shape[2] // 8))
        x = self.enc_fc(x)
        
        return x
    
    def decode(self, x):
        x = self.dec_fc(x)
        x = x.view(-1, 128, self.input_shape[1] // 8, self.input_shape[2] // 8)
        x = torch.relu(self.dec_conv1(x))
        x = self.dec_upsample1(x)
        x = torch.relu(self.dec_conv2(x))
        x = self.dec_upsample2(x)
        x = self.dec_conv3(x)
        
        return x
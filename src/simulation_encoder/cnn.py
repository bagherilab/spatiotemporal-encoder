import torch
import torch.nn as nn

DEBUG = True

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
    def __init__(self, input_shape, out_channels=16, dim_z=100):
        super(ConvolutionalAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.dim_z = dim_z
        self.in_channels = self.input_shape[0]
        self.out_channels = out_channels
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(self.input_shape[0], self.out_channels, kernel_size=3, padding=1)
        self.enc_pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.enc_pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.enc_fc = nn.Linear(self.out_channels * 64 * 64, self.dim_z)
        
        # Decoder
        self.dec_fc = nn.Linear(self.dim_z, self.out_channels * 64 * 64)
        self.dec_conv1 = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.dec_upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.dec_upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv3 = nn.ConvTranspose2d(self.out_channels, self.input_shape[0], kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x):
        debug("Input:", x.size())
        x = torch.relu(self.enc_conv1(x))
        debug("After enc_conv1:", x.size())
        x = self.enc_pool1(x)
        debug("After enc_pool1:", x.size())
        x = torch.relu(self.enc_conv2(x))
        debug("After enc_conv2:", x.size())
        x = self.enc_pool2(x)
        debug("After enc_pool2:", x.size())
        x = torch.relu(self.enc_conv3(x))
        debug("After enc_conv3:", x.size())
        x = x.view(-1, self.out_channels * 64 * 64)
        debug("After view:", x.size())
        x = self.enc_fc(x)
        debug("After enc_fc:", x.size())
        debug("-----------------------")
        return x
    
    def decode(self, x):
        x = self.dec_fc(x)
        ("After dec_fc:", x.size())
        x = x.view(-1, self.out_channels, 64, 64)
        debug("After view:", x.size())
        x = torch.relu(self.dec_conv1(x))
        debug("After dec_conv1:", x.size())
        x = self.dec_upsample1(x)
        debug("After dec_upsample1:", x.size())
        x = torch.relu(self.dec_conv2(x))
        debug("After dec_conv2:", x.size()) 
        x = self.dec_upsample2(x)
        debug("After dec_upsample2:", x.size())
        x = self.dec_conv3(x)
        debug("After dec_conv3:", x.size())
        debug("-----------------------")
        return x
    

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
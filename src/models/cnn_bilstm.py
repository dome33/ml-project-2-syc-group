import torch
import torch.nn as nn
import torch.nn.functional as F

# Credits to : https://github.com/pythonlessons/mltu/blob/main/Tutorials/08_handwriting_recognition_torch/model.py
def activation_layer(activation: str="relu", alpha: float=0.1, inplace: bool=True):
    """ Activation layer wrapper for LeakyReLU and ReLU activation functions

    Args:
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)

    Returns:
        torch.Tensor: activation layer
    """

    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    
    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=alpha, inplace=inplace)


# A block of convolutional layers
class ConvBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        """
        Args:
            in_channels: int, number of input channels.
            out_channels: int, number of output channels.
            kernel_size: int, size of the convolutional kernel.
            stride: int, stride for the convolutional layer.
            padding: int, padding for the convolutional layer.
        """

        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the convolutional block.

        Args:
            x: torch.Tensor, input tensor.

        Returns:
            torch.Tensor, output tensor.
        """
        return self.bn(self.conv(x))


# A residual block with optional skip connection
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, skip_conv=True, stride=1, dropout=0.2, activation="leaky_relu"):
        """
        Args:
            in_channels: int, number of input channels.
            out_channels: int, number of output channels.
            skip_conv: bool, whether to use a skip connection with convolution (default: True).
            stride: int, stride for the first convolutional layer (default: 1).
            dropout: float, dropout rate (default: 0.2).
            activation: str, activation function name (default: 'leaky_relu').
        """

        super(ResidualBlock, self).__init__()
        self.convb1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.act1 = activation_layer(activation)

        self.convb2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(p=dropout)
        
        self.shortcut = None
        if skip_conv:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act2 = activation_layer(activation)
        
    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x: torch.Tensor, input tensor.

        Returns:
            torch.Tensor, output tensor.
        """

        skip = x
        
        out = self.act1(self.convb1(x))
        out = self.convb2(out)

        if self.shortcut is not None:
            out += self.shortcut(skip)

        out = self.act2(out)
        out = self.dropout(out)
        
        return out


# Handwriting recognition network for CTC loss
class CNNBILSTM(nn.Module):

    def __init__(self, num_chars: int, activation: str="leaky_relu", dropout: float=0.2):
        """
        Args:
            num_chars: int, number of output classes (characters).
            activation: str, activation function name (default: 'leaky_relu').
            dropout: float, dropout rate (default: 0.2).
        """

        super(CNNBILSTM, self).__init__()

        # Define residual blocks
        self.rb1 = ResidualBlock(3, 16, skip_conv = True, stride=1, activation=activation, dropout=dropout)
        self.rb2 = ResidualBlock(16, 16, skip_conv = True, stride=2, activation=activation, dropout=dropout)
        self.rb3 = ResidualBlock(16, 16, skip_conv = False, stride=1, activation=activation, dropout=dropout)

        self.rb4 = ResidualBlock(16, 32, skip_conv = True, stride=2, activation=activation, dropout=dropout)
        self.rb5 = ResidualBlock(32, 32, skip_conv = False, stride=1, activation=activation, dropout=dropout)

        self.rb6 = ResidualBlock(32, 64, skip_conv = True, stride=2, activation=activation, dropout=dropout)
        self.rb7 = ResidualBlock(64, 64, skip_conv = True, stride=1, activation=activation, dropout=dropout)

        self.rb8 = ResidualBlock(64, 64, skip_conv = False, stride=1, activation=activation, dropout=dropout)
        self.rb9 = ResidualBlock(64, 64, skip_conv = False, stride=1, activation=activation, dropout=dropout)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(64, 128, bidirectional=True, num_layers=1, batch_first=True)
        self.lstm_dropout = nn.Dropout(p=dropout)

        # Output layer for character predictions
        self.output = nn.Linear(256, num_chars + 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN-BiLSTM model.

        Args:
            images: torch.Tensor, input images in the format (batch, height, width, channels).

        Returns:
            torch.Tensor, predictions for each time step.
        """

        # Normalize images between 0 and 1
        images_flaot = images / 255.0
        
        # Transpose image to channel first
        images_flaot = images_flaot.permute(0, 3, 1, 2)
        
        # Apply convolutions
        x = self.rb1(images_flaot)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.rb9(x)

        x = x.reshape(x.size(0), -1, x.size(1))

        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, 2)

        return x
    
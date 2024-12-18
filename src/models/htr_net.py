import torch.nn as nn
import torch.nn.functional as F


# Configuration class for architecture settings
class arch_cfg: 
    cnn_cfg = [[2, 64], 'M', [3, 128], 'M', [2, 256]]
    head_type = 'both'  # Select from 'both' (rnn + cnn shortcut), 'rnn', 'cnn'
    rnn_type = 'lstm'
    rnn_layers =  3
    rnn_hidden_size =  256
    flattening =  'maxpool'
    stn = False


# Basic block used in CNN for feature extraction
class BasicBlock(nn.Module):
    expansion = 1 # Used for calculating output channel size

    def __init__(self, in_planes, planes, stride=1):
        """
        A residual block with two convolutional layers and a skip connection.

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int): Stride for the first convolution layer.
        """

        super(BasicBlock, self).__init__()

        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection for residual learning
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor after adding the shortcut connection.
        """

        out = F.relu(self.bn1(self.conv1(x)))  # Apply first convolution + activation
        out = self.bn2(self.conv2(out))  # Apply second convolution
        out += self.shortcut(x)  # Add shortcut connection
        out = F.relu(out)  # Apply ReLU activation

        return out



# CNN for feature extraction
class CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        """
        A configurable CNN for feature extraction.

        Args:
            cnn_cfg (list): Configuration for CNN layers (list of blocks and pooling layers).
            flattening (str): Method for flattening feature maps ('maxpool', 'concat', etc.).
        """

        super(CNN, self).__init__()

        self.k = 1
        self.flattening = flattening

        # Initial convolution layer
        self.features = nn.ModuleList([nn.Conv2d(3, 32, 7, [4, 2], 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1

        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(int(m[0])):
                    x = int(m[1])
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x,))
                    in_channels = x
                    cnt += 1

    def forward(self, x, reduce='max'):
        """
        Forward pass through the CNN.

        Args:
            x (Tensor): Input tensor.
            reduce (str): Reduction method for flattening.

        Returns:
            Tensor: Extracted features.
        """

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)
        
        # Flatten the feature map based on the specified method
        if self.flattening=='maxpool':
            fixed_size = 4 
            y = F.max_pool2d(y, kernel_size=(fixed_size, self.k), stride=(fixed_size, 1), padding=(0, self.k//2))
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        return y


# Initialize weights for Conv2D layers
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


# Top CNN-based head for CTC
class CTCtopC(nn.Module):

    def __init__(self, input_size, nclasses, dropout=0.0):
        """
        CTC head with CNN for sequence prediction.

        Args:
            input_size (int): Size of CNN feature maps.
            nclasses (int): Number of output classes.
            dropout (float): Dropout probability.
        """

        super(CTCtopC, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.cnn_top = nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))


    def forward(self, x):
        """
        Forward pass through the CTC head.

        Args:
            x (Tensor): Input feature maps.

        Returns:
            Tensor: Class predictions.
        """

        x = self.dropout(x)

        y = self.cnn_top(x)
        y = y.permute(2, 3, 0, 1)[0]

        return y


# Top RNN-based head for CTC
class CTCtopR(nn.Module):

    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        """
        CTC head with RNN for sequence prediction.

        Args:
            input_size (int): Input feature size.
            rnn_cfg (tuple): RNN configuration (hidden size, number of layers).
            nclasses (int): Number of output classes.
            rnn_type (str): Type of RNN ('gru' or 'lstm').
        """

        super(CTCtopR, self).__init__()

        hidden, num_layers = rnn_cfg

        if rnn_type == 'gru':
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        elif rnn_type == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2) 
        else:
            print('problem! - no such rnn type is defined')
            exit()
        
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))


    def forward(self, x):
        """
        Forward pass through the RNN CTC head.

        Args:
            x (Tensor): Input feature maps.

        Returns:
            Tensor: Class predictions.
        """

        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)

        return y


# Combined CNN + RNN head for CTC
class CTCtopB(nn.Module):

    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        """
        Combined CTC head using both CNN and RNN.

        Args:
            input_size (int): Input feature size.
            rnn_cfg (tuple): RNN configuration.
            nclasses (int): Number of output classes.
            rnn_type (str): Type of RNN ('gru' or 'lstm').
        """

        super(CTCtopB, self).__init__()

        hidden, num_layers = rnn_cfg

        if rnn_type == 'gru':
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        elif rnn_type == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        else:
            print('problem! - no such rnn type is defined')
            exit()
        
        self.fnl = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Sequential(nn.Dropout(.5), 
                                 nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )

    def forward(self, x):
        """
        Forward pass through the combined head.

        Args:
            x (Tensor): Input feature maps.

        Returns:
            Tuple[Tensor, Tensor]: RNN predictions and CNN shortcut predictions.
        """

        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]

        y = self.fnl(y)

        if self.training:
            return y, self.cnn(x).permute(2, 3, 0, 1)[0]
        else:
            return y, self.cnn(x).permute(2, 3, 0, 1)[0]


# Main handwriting recognition model
class HTRNet(nn.Module):

    def __init__(self, nclasses, device = 'cpu'): 
        """
        Handwritten Text Recognition Network.

        Args:
            nclasses (int): Number of output classes.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """

        super(HTRNet, self).__init__()

        self.device = device 
        if arch_cfg.stn: 
            raise NotImplementedError('Spatial Transformer Networks not implemented - you can easily build your own!')
        else:
            self.stn = None

        cnn_cfg = arch_cfg.cnn_cfg
        self.features = CNN(arch_cfg.cnn_cfg, flattening=arch_cfg.flattening).to(device)

        if arch_cfg.flattening=='maxpool' or arch_cfg.flattening=='avgpool':
            hidden = cnn_cfg[-1][-1]
        elif arch_cfg.flattening=='concat':
            hidden = 2 * 8 * cnn_cfg[-1][-1]
        else:
            print('problem! - no such flattening is defined')

        head = arch_cfg.head_type
        if head=='cnn':
            self.top = CTCtopC(hidden, nclasses)
        elif head=='rnn':
            self.top = CTCtopR(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type)
        elif head=='both':
            self.top = CTCtopB(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type)
        self.top = self.top.to(device) 
        
        
    def forward(self, x):
        """
        Forward pass through the HTRNet.

        Args:
            x (Tensor): Input images.

        Returns:
            Tuple[Tensor, Tensor]: RNN and shortcut predictions.
        """

        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
       
        if self.stn is not None:
            x = self.stn(x)
        
        y = self.features(x)
        output, shortcut = self.top(y)
         
        output = F.log_softmax(output, dim=2).permute(1, 0, 2) 
        shortcut = F.log_softmax(shortcut, dim=2).permute(1, 0, 2) 
        return output, shortcut 
    
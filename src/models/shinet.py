
import torch.nn as nn



class ShinetBiLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=64, dropout_p=0.1):
        super(ShinetBiLSTM, self).__init__()

        nc = 1

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
           
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

        input_size = 3 * 512 # height * channels from last conv layer

        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Convolutional layers (CNN)

        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.cnn(x)

        b, c, h, w = x.size()

        # Collapse height and channels into one feature vector per time step
        assert h == 3, "Unexpected height after conv layers"

        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # (B, W, C*H)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Fully connected layer to output character predictions
        out = self.fc(lstm_out)

        prob = out.log_softmax(2)

        return prob




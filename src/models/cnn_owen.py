
import torch.nn as nn



class HandwritingRecognitionCNN_BiLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=64, dropout_p=0.1):
        super(HandwritingRecognitionCNN_BiLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(dropout_p)

        input_size = 4 * 512 # height * channels from last conv layer

        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, dropout=0.1, bidirectional=True, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Convolutional layers (CNN)

        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)

        x = self.conv6(x)

        # Input: x shape = [B, 512, 4, 64] → [batch, channels, height, width]

        # Step 1: Rearrange to put width (time steps) in the middle
        x = x.transpose(1, 3)  # x shape: [B, 64, 4, 512]

        # Step 2: Flatten height and channels to form feature vector per time step
        x = x.reshape(x.size(0), x.size(1), -1)  # x shape: [B, 64, 2048]

        # LSTM
        lstm_out, _ = self.lstm(x)

        #  add another dropout
        lstm_out = self.dropout(lstm_out)

        # Fully connected layer to output character predictions
        out = self.fc(lstm_out)

        prob = out.log_softmax(2)

        return prob




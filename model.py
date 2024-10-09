import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMModel(nn.Module):
    def __init__(self):
        super(ConvLSTMModel, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # 残差连接
        self.residual = nn.Conv2d(64, 512, kernel_size=1, stride=8)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # LSTM 层
        self.lstm = nn.LSTM(input_size=512 * 14 * 14, hidden_size=256, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 3)  # 输出为 angle, torque, speed

    def forward(self, x):
        # 卷积层 + 激活函数
        x = F.relu(self.conv1(x))
        residual = self.residual(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # 残差连接 + Dropout
        x = x + residual
        x = self.dropout(x)

        # 展平并传递到 LSTM
        x = x.view(x.size(0), -1).unsqueeze(1)
        x, (hn, cn) = self.lstm(x)

        # 全连接层
        x = F.relu(self.fc1(hn.squeeze(0)))
        x = self.fc2(x)  # 输出 angle, torque, speed
        return x


net = nn.Sequential(ConvLSTMModel())
X = torch.rand((1,3,224,224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)


import torch.nn as nn
from configs.config import DROPOUT_RATE

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
                      nn.Conv2d(in_channels=3,
                                out_channels=16,
                                kernel_size=3,
                                stride=1,
                                padding=1),
                      nn.BatchNorm2d(16),                # 256 * 256
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size=2,
                                  stride=2)               # 128 * 128
        )

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                              out_channels=32,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(32),
                                      nn.MaxPool2d(kernel_size=2,
                                                    stride=2)            #64 * 64
                              )

        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                              out_channels=64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2,         #32 * 32
                                                    stride=2)
                              )

        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,  # 32 * 32
                                                 stride=2)
                                    )

        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=128,
                                              out_channels=256,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,  # 32 * 32
                                                 stride=2)
                                    )

        self.fc_layer = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                      nn.Linear(256 * 1 * 1, 3))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = self.layer5(out)
      out = self.avg_pool(out)
      out_flatten = out.view(-1, 1 * 1 * 256)
      output = self.fc_layer(out_flatten)
      return output
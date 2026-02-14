import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
P = 0.5
classes = ['paper', 'rock', 'scissors']
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

        self.fc_layer = nn.Sequential(nn.Dropout(P),
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


model = torch.load('model.pt', map_location=torch.device('mps'), weights_only=False)
pre_process = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
cap = cv2.VideoCapture(0)
model.eval()
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame1 = Image.fromarray(frame1)
        frame1 = pre_process(frame1)
        frame1 = frame1.unsqueeze(0)
        frame1 = frame1.to(device)
        result = model(frame1)
        print(result)
        result = torch.softmax(result, dim=1)
        result, index = torch.max(result, dim=1)
        index = int(index)
        result = result.item()
        display_text = f"{classes[index]} {result:.3f}"
        cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
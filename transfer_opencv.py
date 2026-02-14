import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
P = 0.5
classes = ['paper', 'rock', 'scissors']


model = torch.load('/Users/choetaewon/PyCharmMiscProject/model/transfer.pt', map_location=torch.device('mps'), weights_only=False).to(device)
pre_process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
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
        cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
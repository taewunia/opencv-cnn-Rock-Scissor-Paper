from configs.config import CNN_MODEL_PATH, DEVICE, CLASSES, CAP
from utils.data_loader import TEST_PRE_PROCESS
import torch
from models.cnn_model import CNN
from utils.camera import camera
device = torch.device(DEVICE)
model = torch.load(CNN_MODEL_PATH, map_location=device, weight_only=False)

with torch.no_grad():
    model.eval()
    camera(model,CAP, TEST_PRE_PROCESS, device, CLASSES)
from configs.config import CNN_MODEL_PATH, DEVICE, CLASSES
from utils.data_loader import TRAIN_PRE_PROCESS
import torch
from models.cnn_model import CNN
model = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
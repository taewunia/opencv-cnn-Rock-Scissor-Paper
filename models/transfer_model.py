from torchvision import models
import torch.nn as nn

transfer_model = models.convnext_small(weights="IMAGENET1K_V1")
for param in transfer_model.parameters():
    param.requires_grad = False
for param in transfer_model.features[-1].parameters():
    param.requires_grad = True
transfer_model.classifier[2] = nn.Linear(transfer_model.classifier[2].in_features, 3)


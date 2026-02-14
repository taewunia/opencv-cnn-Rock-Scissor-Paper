from torchvision import models, datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torchmetrics import MetricCollection, Accuracy, F1Score
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 2
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)
model = models.convnext_small(weights="IMAGENET1K_V1").to(device)
for param in model.parameters():
    param.requires_grad = False
for param in model.features[-1].parameters():
    param.requires_grad = True
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 3).to(device)
print(model)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.4, 1.0), ratio=(0.75, 1.33)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])
}

train_DS = datasets.ImageFolder(root='/Users/choetaewon/PyCharmMiscProject/datasets/Rock-Paper-Scissors/train',
                               transform=data_transforms['train'],)
test_DS = datasets.ImageFolder(root='/Users/choetaewon/PyCharmMiscProject/datasets/Rock-Paper-Scissors/test',
                               transform=data_transforms['val'],)
train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

metrics = MetricCollection({
    'f1_score':F1Score(task="multiclass",num_classes=3, average='macro'),
    'acc':Accuracy(task="multiclass",num_classes=3)
}).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
val_loos_history = []
for epoch in range(EPOCHS):
    model.train()
    total_val_loss = 0
    avg_val_loss = 0
    train_loading_bar = tqdm(train_DL, desc=f'{epoch+1}/{EPOCHS}', colour='green')
    metrics.reset()
    for x_train_batch, y_train_batch in train_loading_bar:
        x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)
        train_output = model(x_train_batch)
        train_loss = criterion(train_output, y_train_batch)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_preds = torch.argmax(train_output, dim=1)
        metrics.update(train_preds, y_train_batch)
        train_loading_bar.set_postfix(loss=train_loss.item())
    train_result = metrics.compute()
    print(f"{train_result['f1_score'].item()*100:.3f}, {train_result['acc'].item()*100:.3f}")
    test_loading_bar = tqdm(test_DL, desc=f'{epoch + 1}/{EPOCHS}', colour='red')
    metrics.reset()
    model.eval()
    with torch.no_grad():
        for x_val_batch, y_val_batch in test_loading_bar:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
            val_output = model(x_val_batch)
            val_loss = criterion(val_output, y_val_batch)
            val_preds = torch.argmax(val_output, dim=1)
            metrics.update(val_preds, y_val_batch)
            val_loos_history.append(val_loss.item())
            test_loading_bar.set_postfix(loss=val_loss.item())
            total_val_loss += val_loss.item()
        val_result = metrics.compute()
        avg_val_loss = total_val_loss / len(test_DL)
        val_loos_history.append(avg_val_loss)
        print(f"{val_result['f1_score'].item()*100:.3f}, {val_result['acc'].item()*100:.3f}")
        print(f"val_loss: {avg_val_loss:.3f}")


path = '/Users/choetaewon/PyCharmMiscProject/model/transfer.pt'
torch.save(model, path)
print("저장 완료")
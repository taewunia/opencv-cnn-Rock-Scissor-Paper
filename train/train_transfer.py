from utils.data_loader import TRANSFER_PRE_PROCESS
from configs.config import TRAIN_DATE_PATH, TEST_DATE_PATH, DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, CRITERION, OPTIMIZER, NUM_CLASSES, SAVE_PATH, SHUFFLE
from torchmetrics import MetricCollection, Accuracy, F1Score
from tqdm import tqdm
import torch.optim as optim
from models.transfer_model import transfer_model
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = torch.device(DEVICE)
transfer_model = transfer_model()
transfer_model = transfer_model.to(device)
criterion = CRITERION.to(device)
optimizer_class = getattr(optim, OPTIMIZER)
optimizer = optimizer_class(transfer_model.parameters(), lr=LEARNING_RATE)

metrics = MetricCollection({
    'f1_score':F1Score(task="multiclass",num_classes=NUM_CLASSES, average='macro'),
    'acc':Accuracy(task="multiclass",num_classes=NUM_CLASSES)
})

trasnfer_train_DS = datasets.ImageFolder(root=TRAIN_DATE_PATH, transform=TRANSFER_PRE_PROCESS['test'])
trasnfer_test_DS = datasets.ImageFolder(root=TEST_DATE_PATH, transform=TRANSFER_PRE_PROCESS['val'])

train_DL = torch.utills.Dataloader(trasnfer_train_DS, batchsize=BATCH_SIZE, shuffle=SHUFFLE)
test_DL = torch.utills.Dataloader(trasnfer_test_DS, batchsize=BATCH_SIZE, shuffle=False)

val_loos_history = []
for epoch in range(EPOCHS):
    transfer_model.train()
    total_val_loss = 0
    avg_val_loss = 0
    train_loading_bar = tqdm(train_DL, desc=f'{epoch+1}/{EPOCHS}', colour='green')
    metrics.reset()
    for x_train_batch, y_train_batch in train_loading_bar:
        x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)
        train_output = transfer_model(x_train_batch)
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
    transfer_model.eval()
    with torch.no_grad():
        for x_val_batch, y_val_batch in test_loading_bar:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
            val_output = transfer_model(x_val_batch)
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
torch.save(transfer_model, path)
print("저장 완료")
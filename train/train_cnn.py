from utils.data_loader import TRAIN_PRE_PROCESS, TEST_PRE_PROCESS
from configs.config import TRAIN_DATE_PATH, TEST_DATE_PATH, DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, CRITERION, OPTIMIZER, NUM_CLASSES, SAVE_PATH, SHUFFLE
from torchmetrics import MetricCollection, Accuracy, F1Score
from tqdm import tqdm
import torch.optim as optim
from models.cnn_model import CNN
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = torch.device(DEVICE)
CNN_model = CNN()
CNN_model = CNN_model.to(device)
criterion = CRITERION.to(device)
optimizer_class = getattr(optim, OPTIMIZER)
optimizer = optimizer_class(CNN_model.parameters(), lr=LEARNING_RATE)

metrics = MetricCollection({
    'f1_score':F1Score(task="multiclass",num_classes=NUM_CLASSES, average='macro'),
    'acc':Accuracy(task="multiclass",num_classes=NUM_CLASSES)
})

train_DS = datasets.ImageFolder(root=TRAIN_DATE_PATH, transform=TRAIN_PRE_PROCESS)
test_DS = datasets.ImageFolder(root=TEST_DATE_PATH, transform=TEST_PRE_PROCESS)

train_DL = torch.utills.Dataloader(train_DS, batchsize=BATCH_SIZE, shuffle=SHUFFLE)
test_DL = torch.utills.Dataloader(test_DS, batchsize=BATCH_SIZE, shuffle=False)

loss_train_history = []
total_train_loss = 0
loss_test_history = []
total_test_loss = 0

for epochs in range(EPOCHS):
  CNN_model.train()
  train_process_bar = tqdm(train_DL, desc=f"{epochs + 1}/EPOCH", colour='green')
  total_train_loss = 0
  avg_train_loss = 0
  total_test_loss = 0
  avg_test_loss = 0
  metrics.reset()
  for x_batch, y_batch in train_process_bar:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    model_output = CNN_model(x_batch)
    loss = criterion(model_output, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_train_loss += loss.item()
    train_process_bar.set_postfix(loss = loss.item())
    metrics.update(model_output, y_batch)
  train_result = metrics.compute()
  print(f"훈련 정확도:{train_result['acc'].item()*100:.3f} f1스코어:{train_result['f1_score'].item()*100:.3f}")
  avg_train_loss = total_train_loss / len(train_DL)
  loss_train_history.append(avg_train_loss)
  CNN_model.eval()

  test_process_bar = tqdm(test_DL, desc=f"{epochs + 1}/EPOCHS", colour='red')
  metrics.reset()
  with torch.no_grad():
    for x_test_batch, y_test_batch in test_process_bar:
      x_test_batch, y_test_batch = x_test_batch.to(device), y_test_batch.to(device)
      model_test_output = CNN_model(x_test_batch)
      test_loss = criterion(model_test_output, y_test_batch)
      total_test_loss+= test_loss.item()
      test_process_bar.set_postfix(loss = test_loss.item())
      metrics.update(model_test_output, y_test_batch)
    test_result = metrics.compute()
    print(f"실전 정확도:{test_result['acc'].item()*100:.3f} f1스코어:{test_result['f1_score'].item()*100:.3f}")
    avg_test_loss = total_test_loss / len(test_DL)
    print(f"모델 실전 검증[{avg_test_loss:.3f}]")
    print('-' * 20)
    loss_test_history.append(avg_test_loss)
    if test_loss <= 0.4 and test_result['acc'] >= 0.85:
      print("모델 학습 요구치 완료")
      break

plt.plot(range(1, len(loss_train_history) + 1), loss_train_history, label='train loss', color='green')
plt.title("avg_train_loss")
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.legend()
plt.show()

plt.plot(range(1, len(loss_train_history) + 1), loss_test_history, label='test loss', color='red')
plt.title("avg_test_loss")
plt.xlabel("epoch")
plt.ylabel("test_loss")
plt.legend()
plt.show()


CNN_model.eval()
with torch.no_grad():
  x_testbatch, y_testbatch = next(iter(test_DL))
  x_testbatch, y_testbatch = x_testbatch.to(device), y_testbatch.to(device)
  test_output = CNN_model(x_testbatch)
  loss = criterion(test_output, y_testbatch)

  print(f"최종 테스트 결과 loss:{loss:.3f}")

path = SAVE_PATH
user_input = input("모델을 저장하시겠습니까:[Y/N]")

if user_input == 'Y' or user_input == 'y':
  torch.save(CNN_model, path)
  print(f"{path}에 저장완료")

else:
  print("저장하지않았습니다")
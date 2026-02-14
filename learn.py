import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics import MetricCollection, Accuracy, F1Score
from tqdm import tqdm

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"현재 사용중인 기기:{device}")
EPOCH = 70
BATCH_SIZE = 64
LEARNING_RATE = 0.001
P = 0.3

pre_process = torchvision.transforms.Compose([
                                      transforms.Resize(
                                          (256, 256)
                                      ),
                                      transforms.RandomAffine(
                                                              degrees=60 ,
                                                              translate=(0.2, 0.2),
                                                              scale=(0.7, 1.2),
                                                              ),
                                      transforms.ColorJitter(0.2,
                                                             0.2),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.5, 0.5, 0.5],
                                          std=[0.5, 0.5, 0.5]
                                      )
                                      ])

test_pre_process = torchvision.transforms.Compose([
                                      transforms.Resize(
                                          (256, 256)
                                      ),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.5, 0.5, 0.5],
                                          std=[0.5, 0.5, 0.5]
                                      )
])

train_DS = datasets.ImageFolder(root='/Users/choetaewon/PyCharmMiscProject/datasets/Rock-Paper-Scissors/train', transform=pre_process)
test_DS = datasets.ImageFolder(root='/Users/choetaewon/PyCharmMiscProject/datasets/Rock-Paper-Scissors/test', transform=test_pre_process)

train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=False)


metrics = MetricCollection({
    'f1_score':F1Score(task="multiclass",num_classes=3, average='macro'),
    'acc':Accuracy(task="multiclass",num_classes=3)
})
metric = metrics.to(device)
print(train_DS)
print(train_DL)

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


model = CNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_train_history = []
total_train_loss = 0
loss_test_history = []
total_test_loss = 0

for epochs in range(EPOCH):
  model.train()
  train_process_bar = tqdm(train_DL, desc=f"{epochs + 1}/EPOCH", colour='green')
  total_train_loss = 0
  avg_train_loss = 0
  total_test_loss = 0
  avg_test_loss = 0
  metrics.reset()
  for x_batch, y_batch in train_process_bar:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    model_output = model(x_batch)
    loss = criterion(model_output, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_train_loss += loss.item()
    train_process_bar.set_postfix(loss = loss.item())
    metric.update(model_output, y_batch)
  train_result = metrics.compute()
  print(f"훈련 정확도:{train_result['acc'].item()*100:.3f} f1스코어:{train_result['f1_score'].item()*100:.3f}")
  avg_train_loss = total_train_loss / len(train_DL)
  loss_train_history.append(avg_train_loss)
  model.eval()

  test_process_bar = tqdm(test_DL, desc=f"{epochs + 1}/EPOCH", colour='red')
  metrics.reset()
  with torch.no_grad():
    for x_test_batch, y_test_batch in test_process_bar:
      x_test_batch, y_test_batch = x_test_batch.to(device), y_test_batch.to(device)
      model_test_output = model(x_test_batch)
      test_loss = criterion(model_test_output, y_test_batch)
      total_test_loss+= test_loss.item()
      test_process_bar.set_postfix(loss = test_loss.item())
      metric.update(model_test_output, y_test_batch)
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


model.eval()
with torch.no_grad():
  x_testbatch, y_testbatch = next(iter(test_DL))
  x_testbatch, y_testbatch = x_testbatch.to(device), y_testbatch.to(device)
  test_output = model(x_testbatch)
  loss = criterion(test_output, y_testbatch)

  print(f"최종 테스트 결과 loss:{loss:.3f}")

path = '/Users/choetaewon/PyCharmMiscProject/model.pt'
user_input = input("모델을 저장하시겠습니까:[Y/N]")

if user_input == 'Y' or user_input == 'y':
  torch.save(model, path)
  print(f"{path}에 저장완료")

else:
  print("저장하지않았습니다")
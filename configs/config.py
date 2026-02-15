#model training
import torch.nn as nn
EPOCHS = 70
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = "Adam"
SHUFFLE = True

#model
DEVICE = "mps"
NUM_CLASSES = 3
SAVE_PATH = ""
CLASSES = ['paper', 'rock', 'scissors']

#model path
CNN_MODEL_PATH = "/Users/choetaewon/PyCharmMiscProject/model.pt"
TRANSFER_MODEL_PATH = "/Users/choetaewon/PyCharmMiscProject/model/transfer.pt"

#data path
TRAIN_DATE_PATH = "/Users/choetaewon/PyCharmMiscProject/datasets/Rock-Paper-Scissors/train"
TEST_DATE_PATH = "/Users/choetaewon/PyCharmMiscProject/datasets/Rock-Paper-Scissors/test"

#cap
CAP = 0
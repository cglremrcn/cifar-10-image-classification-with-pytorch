import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data import get_data
from model import get_resnet18  # Burada model.py'dan import ediyoruz
from utils import accuracy_fn
from config import LR,BATCH_SIZE,EPOCHS
from utils import train,test_step,train_step

# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# get the data
train_data, test_data = get_data()
train_dataloader = DataLoader(train_data, batch_size= BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size= BATCH_SIZE, shuffle=False)


# get the model
model_0 = get_resnet18(10,pretrained=True).to(device)

# set optimizer and loss func
optimizer = torch.optim.Adam(model_0.parameters(),lr = LR)
loss_fn = nn.CrossEntropyLoss()

results = train(model_0, train_dataloader, test_dataloader, optimizer, loss_fn, device = device , epochs = EPOCHS)

# save the model when train ends
torch.save(model_0.state_dict(), "saved_models/resnet18_cifar10.pth")

print("Model saved: saved_models/resnet18_cifar10.pth")
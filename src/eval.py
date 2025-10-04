from model import get_resnet18
from data import get_data
from utils import test_step
import torch
from torch import nn
from config import BATCH_SIZE
from torch.utils.data import DataLoader

# set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = get_resnet18(num_classes=10).to(device)
model.load_state_dict(torch.load("saved_models/resnet18_cifar10.pth", map_location=device,weights_only= True))

# test loader
_, test_data= get_data()

test_dataloader = DataLoader(test_data, batch_size= BATCH_SIZE, shuffle=False)

# set the loss func
loss_fn = nn.CrossEntropyLoss()

# evaluate
test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

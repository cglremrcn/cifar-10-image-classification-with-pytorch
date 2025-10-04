import torch
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from data import get_data
from model import get_resnet18
from utils import make_predictions
from timeit import default_timer as timer

# set the device
device = "cuda" if torch.cuda.is_available() else "cpu"


train_data, test_data = get_data()
class_names = test_data.classes

# load model
model = get_resnet18(num_classes=10).to(device)
model.load_state_dict(torch.load("saved_models/resnet18_cifar10.pth", 
                                 map_location=device, 
                                 weights_only=True))

# make predictions
pred_time_start = timer()
model.eval()
y_pred_tensor = make_predictions(model, test_data, device)
pred_time_end = timer()
print(f'Prediction time {pred_time_end - pred_time_start}')

y_pred_labels = torch.argmax(y_pred_tensor, dim=1)


y_true = torch.tensor([label for _, label in test_data])
y_true = y_true.to(y_pred_labels.device)


# conf mat from torchmetrics
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_labels, target=y_true)

# plot conf matrix with mlxtend
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.cpu().numpy(),
    class_names=class_names,
    figsize=(10,7)
)
plt.show()

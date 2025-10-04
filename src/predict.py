import torch
from model import get_resnet18
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


# set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model
model = get_resnet18(num_classes=10).to(device)
model.load_state_dict(torch.load("saved_models/resnet18_cifar10.pth", map_location=device,weights_only= True))


classes = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

img_path = "data/cat.jpg"
image = Image.open(img_path)

# get the size for cifar - 10
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

image = transform(image).unsqueeze(0).to(device)

# predict on a single image
model.eval()

with torch.inference_mode():
    outputs = model(image)                  
    probs = F.softmax(outputs, dim=1)       #
    max_prob, pred_idx = torch.max(probs, dim=1)
    pred_class = classes[pred_idx] if max_prob.item() > 0.5 else None  



plt.imshow(image.squeeze(0).cpu().permute(1,2,0))
plt.axis('off')
plt.title(f"Predicted: {pred_class}")
plt.show()

from pathlib import Path
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
data_path = Path('data/')

image_path = data_path / 'cifar10_images'

if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one")
    image_path.mkdir(parents=True, exist_ok=True)



train_dir = image_path / "train"
test_dir = image_path / "test"
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),   
    transforms.RandomRotation(10),       
    transforms.ToTensor(),               
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

print(train_dir, test_dir)

train_data = datasets.CIFAR10(root  = train_dir,train = True, download=True,transform= transform_train,target_transform=None)
test_data = datasets.CIFAR10(root  = test_dir,train = False, download=True,transform= ToTensor(),target_transform=None)

def get_data():
    return train_data,test_data
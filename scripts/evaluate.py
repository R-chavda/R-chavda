import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

MODEL_PATH = 'models/fabric_resnet18.pth'
VAL_DIR = 'data/validation'
BATCH_SIZE = 32

def load_model(num_classes):
    try:
        model = models.resnet18(pretrained=True)
    except Exception:
        model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def evaluate():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = load_model(len(val_dataset.classes))
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0
    print(f'Validation Accuracy: {acc:.4f}')

if __name__ == '__main__':
    evaluate()

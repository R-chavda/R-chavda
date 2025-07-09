import torch
from torchvision import transforms, models
from PIL import Image
import os
import sys

MODEL_PATH = 'models/fabric_resnet18.pth'
CLASS_NAMES = sorted(os.listdir('data/train'))  # Automatically get updated class names

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

def predict(image_path, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/inference.py <image_path>')
        sys.exit(1)
    image_path = sys.argv[1]
    model = load_model(len(CLASS_NAMES))
    pred = predict(image_path, model, CLASS_NAMES)
    print(f'Predicted class: {pred}')

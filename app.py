import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os

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

def predict(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

st.title('Fabric Defect Classification')
st.write('Upload a fabric image to detect defects in real time.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    with st.spinner('Classifying...'):
        model = load_model(len(CLASS_NAMES))
        pred = predict(image, model, CLASS_NAMES)
    st.write(f'Prediction: **{pred}**')


import streamlit as st
import requests
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

# Define CNN model
class PenguinCNN(nn.Module):
    def __init__(self, num_classes):
        super(PenguinCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define class names
class_names = ['Chinstrap Penguins', 'Piplup', 'Adelie Penguins', 'Gentoo Penguins', 'Miniso Penguins']
class_to_index= {class_name: index for index, class_name in enumerate(class_names)}

model = PenguinCNN(num_classes=len(class_names))

# Load quantized model state dictionary
# quantized_state_dict = torch.load('quantized_penguin_model.pth', map_location=torch.device('cpu'))

# Assign the quantized state dictionary to the model
# model.load_state_dict(quantized_state_dict)

short_quantized_model = torch.load('short_quantized_penguin_model.pth')
model.eval()

# Function to make predictions
def predict_species(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class_index = predicted.item()
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Streamlit app
def main():
    st.title('Penguin Image Classifier')
    uploaded_file = st.file_uploader('Upload an image of a penguin', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write('')
        st.write('Classifying...')

        # Make a prediction
        species = predict_species(uploaded_file)
        st.write(f'Prediction: {species}')

if __name__ == '__main__':
    main()

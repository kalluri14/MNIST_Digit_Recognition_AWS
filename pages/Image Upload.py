import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

# Define the DigitClassifierCNN model (matching the training configuration)
class DigitClassifierCNN(torch.nn.Module):
    def __init__(self):
        super(DigitClassifierCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(16 * 7 * 7, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model

model_path = "final_mnist_digit_classifier.pth"
model = DigitClassifierCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define transformation to match training preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure image is in grayscale
    transforms.Resize((28, 28)),  # Resize to match training input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize using the same values as training
])

def predict_image(image, model):
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, 1, H, W)
    
    # Make the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Streamlit app
st.title("Handwritten Digit Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "gif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale if needed
    st.image(image, caption='Uploaded Image', width=150)  # You can adjust the width value as per your preference
    st.write("")

    # Predict
    predicted_class = predict_image(image, model)
    st.write(f'Predicted class: {predicted_class}')


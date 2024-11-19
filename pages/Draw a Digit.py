import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import os

# Load your trained model here
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

# Center alignment of the whole content
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.title("Handwritten Digit Classifier")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgb(255, 255, 255)",  # Background color
    stroke_width=10,
    stroke_color="rgb(0, 0, 0)",
    background_color="rgb(255, 255, 255)",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert image to grayscale and check if there is meaningful drawing
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    img_array = np.array(img)
    # Count non-white pixels (assuming white is 255)
    non_white_pixels = np.sum(img_array < 255)

    if non_white_pixels > 50:  # Adjust this threshold based on your requirement
        img = ImageOps.invert(img)  # Invert the colors (white background, black digit)
        img = img.resize((28, 28))  # Resize to 28x28 pixels
        img = transforms.ToTensor()(img).unsqueeze(0)  # Add batch dimension and convert to tensor
        img = transforms.Normalize((0.5,), (0.5,))(img)  # Normalize

        # Make prediction
        with torch.no_grad():
            output = model(img)
            _, predicted_class = torch.max(output, 1)
            st.markdown(f"<h2>Predicted class: {predicted_class.item()}</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>Predicted class: None</h2>", unsafe_allow_html=True)  # Display "None" if no meaningful drawing is detected
else:
    st.markdown("<h2>Predicted class: None</h2>", unsafe_allow_html=True)  # Display "None" if the canvas is empty

st.markdown("</div>", unsafe_allow_html=True)

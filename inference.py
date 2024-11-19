import torch
import os
import tarfile
import boto3
from torchvision import transforms
from PIL import Image
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# Define the model architecture (same as used during training)
class DigitClassifierCNN(torch.nn.Module):
    def __init__(self):
        super(DigitClassifierCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(16 * 7 * 7, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.pool = torch.nn.MaxPool2d(2, 2)
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

# Load the model
def model_fn(model_dir):
    logger.info("Starting model_fn.")
    try:
      
        s3 = boto3.client('s3')
        bucket_name = 'mnistdataset'  
        s3_key = 'images/output/pytorch-training-2024-11-12-02-18-34-727/output/model.tar.gz '  # Replace with your key
        local_tar_path = os.path.join(model_dir, 'model.tar.gz')

        logger.info(f"Downloading model from S3 bucket {bucket_name} with key {s3_key}.")
        s3.download_file(bucket_name, s3_key, local_tar_path)

        # Extract the model.tar.gz file
        extracted_dir = os.path.join(model_dir, 'extracted_model')
        if not os.path.exists(extracted_dir):
            os.makedirs(extracted_dir)
        logger.info("Extracting model tar file.")
        with tarfile.open(local_tar_path, 'r:gz') as tar:
            tar.extractall(path=extracted_dir)

        # Load the extracted model file (assuming it's named 'final_mnist_digit_classifier.pth')
        model_path = os.path.join(extracted_dir, 'final_mnist_digit_classifier.pth')
        logger.info(f"Loading model from {model_path}.")
        model = DigitClassifierCNN()  # Ensure the model architecture matches what was used during training
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error in model_fn: {str(e)}")
        raise

# Preprocess input data
def input_fn(request_body, request_content_type):
    logger.info(f"input_fn called with content type: {request_content_type}")
    try:
        if request_content_type == 'application/json':
            data = json.loads(request_body)
            image_path = data.get('image_path')
            logger.info(f"Received image path: {image_path}")
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),  # Ensure consistent size
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            image = transform(image).unsqueeze(0)  # Add batch dimension
            logger.info("Image preprocessed successfully.")
            return image
        else:
            logger.error(f"Unsupported content type: {request_content_type}")
            raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise

# Perform prediction
def predict_fn(input_data, model):
    logger.info("Starting prediction.")
    try:
        with torch.no_grad():
            output = model(input_data)
            _, predicted_class = torch.max(output, 1)
        logger.info(f"Prediction completed. Predicted class: {predicted_class.item()}")
        return predicted_class.item()
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise

# Format the output
def output_fn(prediction, content_type):
    logger.info(f"Formatting output for content type: {content_type}")
    try:
        if content_type == 'application/json':
            return json.dumps({'predicted_class': prediction})
        else:
            logger.error(f"Unsupported content type for output: {content_type}")
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        raise

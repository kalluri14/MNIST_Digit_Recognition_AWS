import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
import argparse
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Define Custom Dataset
class CustomMNISTDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        self.images_dir = images_dir
        if os.path.isdir(labels_csv):
            labels_csv = os.path.join(labels_csv, next(f for f in os.listdir(labels_csv) if f.endswith('.csv')))
        self.labels = pd.read_csv(labels_csv, header=None).values.flatten()
        self.image_files = sorted(os.listdir(images_dir), key=lambda x: int(os.path.splitext(x)[0]))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the Model
class DigitClassifierCNN(nn.Module):
    def __init__(self):
        super(DigitClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(args):
    # Hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    # Transformations for images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    train_dataset = CustomMNISTDataset(args.train_images_dir, args.train_labels_csv, transform=transform)
    test_dataset = CustomMNISTDataset(args.test_images_dir, args.test_labels_csv, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = DigitClassifierCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print training loss in a SageMaker-compatible format
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}")
        # SageMaker metric logging
        print(f"Training Loss: {avg_loss:.4f}")

    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    # Print validation accuracy in a SageMaker-compatible format
    print(f"Validation Accuracy: {accuracy:.2f}")
    logger.info(f"Validation Accuracy: {accuracy:.2f}")

    # Save the model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'final_mnist_digit_classifier.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--train-images-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN_IMAGES'])
    parser.add_argument('--train-labels-csv', type=str, default=os.environ['SM_CHANNEL_TRAIN_LABELS'])
    parser.add_argument('--test-images-dir', type=str, default=os.environ['SM_CHANNEL_TEST_IMAGES'])
    parser.add_argument('--test-labels-csv', type=str, default=os.environ['SM_CHANNEL_TEST_LABELS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()
    train(args)


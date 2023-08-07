
# Install libraries
! pip install torch torchvision streamlit pillow

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# Define CNN model
class PenguinCNN(nn.Module):
    def __init__(self, num_classes): #edit
        super(PenguinCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #3 inchannels, 32 outchannels
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


# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Data normalization for testing
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# Load the dataset
data_dir = 'data'
class_names = os.listdir(data_dir)
# print(class_names)
class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
# print(class_to_index)
image_paths = []
labels = []

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image_paths.append(image_path)
            labels.append(class_to_index[class_name])

# Train test split
train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size = 0.3, random_state=42)

class PenguinDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        # print(label)

        if self.transform:
            image = self.transform(image)

        return image, label

train_dataset = PenguinDataset(train_image_paths, train_labels, transform=transform_train)
test_dataset = PenguinDataset(test_image_paths, test_labels, transform=transform_test)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Initialize the model and loss function, optimizer
device = torch.device('cuda')
num_classes = len(set(train_labels))
model = PenguinCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_loss = []
train_acc = []
test_loss = []
test_acc = []

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_running_loss = 0.0
    train_correct_predictions = 0
    train_total_samples = 0

    # Train set
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()

        # Compute the number of correct predictions
        _, predicted = torch.max(outputs, 1)
        train_correct_predictions += (predicted == labels).sum().item()
        train_total_samples += labels.size(0)

    train_epoch_loss = train_running_loss / len(train_loader)
    train_loss.append(train_epoch_loss)
    train_epoch_accuracy = train_correct_predictions / train_total_samples
    train_acc.append(train_epoch_accuracy)

    model.eval()
    test_running_loss = 0.0
    test_correct_predictions = 0
    test_total_samples = 0

    # Test set
    with torch.no_grad():
      for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_running_loss += loss.item()

        # Compute the number of correct predictions
        _, predicted = torch.max(outputs, 1)
        test_correct_predictions += (predicted == labels).sum().item()
        test_total_samples += labels.size(0)

    test_epoch_loss = test_running_loss / len(test_loader)
    test_loss.append(test_epoch_loss)
    test_epoch_accuracy = test_correct_predictions / test_total_samples
    test_acc.append(test_epoch_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_epoch_loss:.4f}, Accuracy: {train_epoch_accuracy:.4f}')


# Save the trained model
torch.save(model.state_dict(), 'penguin_classifier.pth')

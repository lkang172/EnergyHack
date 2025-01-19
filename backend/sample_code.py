#Sample 1

'''
self.loss_fn = nn.CrossEntropyLoss()

def createVisualModel(self):
    self.visualModel = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.Flatten()
    )

def createAudioModel(self):
    self.audioModel = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.Flatten()
    )

def createFusionModel(self):
    self.fusionModel = nn.Sequential(
        nn.Linear(59648, 2048),  
        nn.ReLU(),
        nn.BatchNorm1d(2048),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256)
    )

def createFCModel(self):
    self.fcModel = nn.Sequential(
        nn.Linear(29824, 1024),
        nn.ReLU(),
        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
'''

# Sample 2
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
input_size = 784  # 28x28 pixels
hidden_size = 128
num_classes = 10  # Digits 0-9
batch_size = 64
learning_rate = 0.01
num_epochs = 5

# Define a simple feedforward neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.Linear(59648, 2048),  
        nn.ReLU(),
        nn.BatchNorm1d(2048),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the network, loss function, and optimizer
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Training the model...")
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Flatten the images to (batch_size, 784)
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Testing loop
print("\nTesting the model...")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
'''

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
input_size = 784  # 28x28 pixels (flattened)
hidden_layer_size = 128
number_of_classes = 10  # Digits 0-9
mini_batch_size = 64
learning_rate_value = 0.01
number_of_epochs = 5

# Define a simple feedforward neural network
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_features, hidden_units, output_classes):
        super(SimpleNeuralNetwork, self).__init__()
        self.first_fully_connected_layer = nn.Linear(input_features, hidden_units)
        self.activation_function = nn.ReLU()
        self.second_fully_connected_layer = nn.Linear(hidden_units, output_classes)

    def forward(self, input_data):
        hidden_output = self.first_fully_connected_layer(input_data)
        activated_output = self.activation_function(hidden_output)
        final_output = self.second_fully_connected_layer(activated_output)
        return final_output

# Load MNIST dataset
data_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

training_dataset = datasets.MNIST(root='./data', train=True, transform=data_transformation, download=True)
testing_dataset = datasets.MNIST(root='./data', train=False, transform=data_transformation, download=True)

training_data_loader = DataLoader(dataset=training_dataset, batch_size=mini_batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=testing_dataset, batch_size=mini_batch_size, shuffle=False)

# Initialize the network, loss function, and optimizer
device_to_use = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
neural_network_model = SimpleNeuralNetwork(input_size, hidden_layer_size, number_of_classes).to(device_to_use)
loss_function = nn.CrossEntropyLoss()
optimizer_function = optim.Adam(neural_network_model.parameters(), lr=learning_rate_value)

# Training loop
print("Training the model...")
for epoch in range(number_of_epochs):
    neural_network_model.train()
    for batch_index, (image_batch, label_batch) in enumerate(training_data_loader):
        # Flatten the images to (mini_batch_size, 784)
        flattened_images = image_batch.view(-1, input_size).to(device_to_use)
        labels = label_batch.to(device_to_use)

        # Forward pass
        predictions = neural_network_model(flattened_images)
        computed_loss = loss_function(predictions, labels)

        # Backward pass and optimization
        optimizer_function.zero_grad()
        computed_loss.backward()
        optimizer_function.step()

        if (batch_index + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{number_of_epochs}], Step [{batch_index + 1}/{len(training_data_loader)}], Loss: {computed_loss.item():.4f}")

# Testing loop
print("\nTesting the model...")
neural_network_model.eval()
with torch.no_grad():
    total_correct_predictions = 0
    total_test_samples = 0
    for image_batch, label_batch in testing_data_loader:
        flattened_images = image_batch.view(-1, input_size).to(device_to_use)
        labels = label_batch.to(device_to_use)

        predictions = neural_network_model(flattened_images)
        _, predicted_classes = torch.max(predictions, 1)
        total_test_samples += labels.size(0)
        total_correct_predictions += (predicted_classes == labels).sum().item()

    test_accuracy_percentage = 100 * total_correct_predictions / total_test_samples
    print(f"Test Accuracy: {test_accuracy_percentage:.2f}%")

'''
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_classes = 10  # Digits 0-9
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Define a 2D CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  # Output: 32x28x28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),  # Output: 32x14x14

            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # Output: 64x14x14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),  # Output: 64x7x7

            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # Output: 128x7x7
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2)  # Output: 128x3x3
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),  # Output: 256
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),  # Output: 128
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output: 10 (num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the network, loss function, and optimizer
model = CNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Training the model...")
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # Shape: [batch_size, 1, 28, 28]
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Testing loop
print("Testing the model...")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
'''
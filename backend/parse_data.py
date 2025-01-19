import ast

source_code = """
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

# Testing loop
print("Testing the model...")
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
    print(f"Test Accuracy:%")
"""
    

def parse_function(source_code):
    layerToInt = {"Linear": 0, "Conv1d": 1, "Conv2d": 2, "Conv3d": 3, "MaxPool1d": 4, "MaxPool2d": 5, "MaxPool3d": 6, "AvgPool1d": 7, "AvgPool2d": 8, "AvgPool3d": 9, 
            "RNN": 10, "LSTM" : 11, "GRU": 12, "ReLU": 13, "Sigmoid" : 14, "Tanh" : 15, "BatchNorm1d": 16, "BatchNorm2d": 17, "LayerNorm": 18,
            "Dropout": 19, "Dropout2d": 20, "Dropout3d": 21, "flatten": 22, "Embedding": 23, "CrossEntropyLoss": 24, "MSELoss": 25, "SmoothL1Loss": 26, 
            "NLLLoss": 27, "HingeEmbeddingLoss": 28, "KLDivLoss": 29, "BCELoss": 30}

    intToParams = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 
                18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: []}

    tree = ast.parse(source_code)

    class ArgVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                # print(f"    Method Call: {node.func.attr}")
                layer_name = node.func.attr
                if layer_name in layerToInt:
                    sub_array = []
                    for arg in node.args:
                        if isinstance(arg, ast.Constant):
                            sub_array.append(arg.value)
                        elif isinstance(arg, ast.Name):
                            sub_array.append(arg.id)
                    for keyword in node.keywords:
                        if isinstance(arg, ast.Constant):
                            sub_array.append(keyword.value.value)
                        elif isinstance(arg, ast.Name):
                            sub_array.append(keyword.value.value)
                    intToParams[layerToInt[layer_name]].append(sub_array)
            self.generic_visit(node) 


    visitor = ArgVisitor()
    visitor.visit(tree)

    return intToParams

# parse_function(source_code)